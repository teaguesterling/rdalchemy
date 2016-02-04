#!/usr/bin/env python
#
# Copyright (C) 2014 Teague Sterling, John Irwin,
# Regents of the University of California
#
# This implementation utilizes code and methods from Riccardo Vianello
# as well as code structure and inspiration from geoalchemy2

import base64
import contextlib
import functools
import numbers
import operator
import string
import types

import numpy as np

from sqlalchemy import event, Table, bindparam, func
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.sql import expression, functions, type_coerce, elements, operators
from sqlalchemy.types import (
    UserDefinedType, 
    _Binary, 
    TypeDecorator, 
    BINARY,
    Float,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql.base import (
    DOUBLE_PRECISION,
    ischema_names,
)

from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdchem
from rdkit.Chem.Draw import MolToImage
from rdkit.Chem.rdDistGeom import EmbedMolecule

_all_bytes = string.maketrans('', '')


class ChemistryError(ValueError):
    pass


class CustomEqualityBinaryExpression_HACK(elements.BinaryExpression):
    custom_opstrings = ()

    def __bool__(self):
        try:
            return super(CustomEqualityBinaryExpression_HACK, self).__bool__()
        except TypeError:
            if self.operator.opstring in self.custom_opstrings:
                return operators.eq(hash(self._orig[0]), hash(self._orig[1]))
            else:
                raise

    __nonzero__ = __bool__

    @classmethod
    def _override_expr(cls, source, opstrings):
        """Create a shallow copy of this ClauseElement.
        This method may be used by a generative API.  Its also used as
        part of the "deep" copy afforded by a traversal that combines
        the _copy_internals() method.
        """
        c = cls.__new__(cls)
        c.__dict__ = source.__dict__.copy()
        elements.ClauseElement._cloned_set._reset(c)
        elements.ColumnElement.comparator._reset(c)

        # this is a marker that helps to "equate" clauses to each other
        # when a Select returns its list of FROM clauses.  the cloning
        # process leaves around a lot of remnants of the previous clause
        # typically in the form of column expressions still attached to the
        # old table.
        c._is_clone_of = source

        c.custom_opstrings = opstrings

        return c


def _remove_control_characters(data):
    if not isinstance(data, basestring):
        raise ValueError("Data must be a string")
    else:
        data = str(data)
    return data.translate(_all_bytes, _all_bytes[:32])


## Datatype Converstions
## TODO: This should be reorganized... do we even want 
##       automatic conversion from ctab to mol?
## Break these out into separate files for each type


## Mol conversions ##################################################

def ensure_mol(mol, sanitize=True):
    if not isinstance(mol, (Chem.Mol, rdchem.Mol)):
        raise ValueError("Not already an instance of rdkit.Chem.Mol")
    if sanitize:
        try:
            Chem.SanitizeMol(mol)
        except ValueError as e:
           raise ChemistryError(str(e))
    return mol

def extract_mol_element(mol, sanitize=True):
    if hasattr(mol, 'as_mol'):
        mol = mol.as_mol
    elif hasattr(mol, 'mol'):
        mol = mol.mol
    else:
        raise ValueError("Not an instance of RawMolElement or compatible")
    if sanitize:
        try:
            Chem.SanitizeMol(mol)
        except ValueError as e:
            raise ChemistryError(str(e))
    return mol

def smiles_to_mol(smiles, sanitize=True):
    smiles = _remove_control_characters(smiles)
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        raise ValueError("Failed to parse SMILES: `{0}`".format(smiles))
    if sanitize:
        try:
            Chem.SanitizeMol(mol)
            Chem.AssignStereochemistry(mol)
        except ValueError as e:
            raise ChemistryError(str(e))
    return mol

def smarts_to_mol(smarts, sanitize=True):
    smiles = _remove_control_characters(smarts)
    mol = Chem.MolFromSmarts(smarts, mergeHs=True)
    if mol is None:
        raise ValueError("Failed to parse SMARTS: `{0}`".format(smarts))
    if sanitize:
        Chem.SanitizeMol(mol, catchErrors=True)
        Chem.AssignStereochemistry(mol)
    return mol

def binary_to_mol(data, sanitize=True):
    try:
        mol = Chem.Mol(data)
    except Exception:  # This is a proxy for Boost.Python.ArgumentError
        raise ValueError("Invalid binary mol data: `{0}`".format(data))
    if sanitize:
        try:
            Chem.SanitizeMol(mol)
            Chem.AssignStereochemistry(mol)
        except ValueError as e:
            raise ChemistryError(str(e))
    return mol

def ctab_to_mol(data, sanitize=True):
    smiles = _remove_control_characters(data)
    mol = Chem.MolFromMolBlock(data, sanitize=sanitize, removeHs=sanitize)
    if mol is None:
        raise ValueError("Failed to parse CTAB")
    if sanitize:
        try:
            Chem.SanitizeMol(mol)
            Chem.AssignStereochemistry(mol)
        except ValueError as e:
            raise ChemistryError(str(e))
    return mol

def inchi_to_mol(inchi, sanitize=True):
    smiles = _remove_control_characters(inchi)
    mol = Chem.MolFromInchi(inchi, sanitize=sanitize, removeHs=sanitize)
    if mol is None:
        raise ValueError("Failed to parse InChI: `{0}`".format(inchi))
    if sanitize:
        try:
            Chem.SanitizeMol(mol)
            Chem.AssignStereochemistry(mol)
        except ValueError as e:
            raise ChemistryError(str(e))
    return mol

# Want to maintain order
MOL_PARSERS = [
    ('mol', ensure_mol),
    ('element', extract_mol_element),
    ('binary', binary_to_mol),
    ('smiles', smiles_to_mol),
    ('smarts', smarts_to_mol),
    ('ctab', ctab_to_mol),
    ('inchi', inchi_to_mol),
]

def attempt_mol_coersion(data, sanitize=True, exclude=()):
     # RDKit doesn't like Unicode
    if isinstance(data, (basestring, buffer)):
        data = str(data)
    
    # Record all parsing errors
    errors = []
    
    # Try all known mol parsers
    for fmt, parser in MOL_PARSERS:
        if fmt in exclude:
          errors.append("Explicitly skipping {}".format(fmt))
          continue
        try:
            mol = parser(data, sanitize=sanitize)
            return fmt, mol
        except ChemistryError as error:
            errors.append(str(error))
            break
        except ValueError as error:
            errors.append(str(error))
    raise ValueError("Failed to convert `{0}` to mol. Errors were: {1}".format(data, ", ".join(errors)))

def coerce_to_mol(data, sanitize=True, exclude=()):
    fmt, mol = attempt_mol_coersion(data, sanitize=sanitize, exclude=exclude)
    return mol
    
def infer_mol_format(data, sanitize=True, exclude=()):
    fmt, mol = attempt_mol_coersion(data, sanitize=sanitize, exclude=exclude)
    return fmt


## BFP Conversions ##################################################

def chunks(xs, k):
    n = len(xs)
    for i in xrange(0, n, k):
        yield xs[i:i+k]
        
def byte_from_hex(value):
    return int(value, 16)

def byte_to_hex(value):
    return "{:02x}".format(value)


def ensure_bfp(value, size=None):
    if not isinstance(value, DataStructs.ExplicitBitVect):
        raise ValueError("Not already a bfp (rdkit.DataStructs.ExplicitBitVect)")
    if size is not None and size != value.GetNumBits():
        raise Exception("BFP size does not match expected {0}".format(size))
    return value


def extract_bfp_element(value, size=None):
    if hasattr(value, 'as_bfp'):
        value = value.as_bfp
    else:
        raise ValueError("Not already a bfp element (or compatable)")
    if size is not None and size != value.GetNumBits():
        raise ValueError("BFP size does not match expected {0}".format(size))
    return value
    

def bfp_from_raw_binary_text(raw, size=None):
    vect = DataStructs.CreateFromBinaryText(raw)
    if vect.GetNumBits() != size:
        raise ValueError("BFP size does not match expected {0}".format(size))
    return vect

def bfp_to_raw_binary_text(bfp):
    return bfp.ToBinary()
    
def bytes_from_binary_text(binary_text):
    if not isinstance(binary_text, basestring):
        raise ValueError("Binary text must be a string")
    if binary_text.startswith(r'\x'):
        binary_text = binary_text[2:]
    else:
        raise ValueError("Binary text must be hex-encoded and prefixed with '\\x'")
    byte_chunks = list(chunks(binary_text, 2))
    byte_values = map(byte_from_hex, byte_chunks)
    values = np.array(byte_values, dtype=np.uint8)
    return values

def bytes_to_binary_text(byte_values):
    hex_chars = map(byte_to_hex, byte_values)
    binary_text = r'\x' + ''.join(hex_chars)
    return binary_text

def bytes_from_chars(chars):
    byte_values = map(ord, chars)
    values = np.array(byte_values, dtype=np.uint8)
    return values

def bytes_to_chars(byte_values):
    char_values = map(chr, byte_values)
    chars = ''.join(char_values)
    return chars

def bfp_from_bits(bits, size=None):
    if size is None:
        raise ValueError("Cannot create BFP from on-bit list without explicit size")
    if not all(isinstance(idx, (numbers.Integral, np.integer, int)) for idx in bits):
        try:
            bits = map(int, bits)
        except ValueError:
            raise ValueError("Can only create BFP from collection of integers")
    vect = DataStructs.ExplicitBitVect(size)
    on_bits = np.nonzero(bits)[0]
    vect.SetBitsFromList(on_bits)
    return vect

def bfp_to_bits(vect):
    num_bits = vect.GetNumBits()
    bits = np.array(num_bits, dtype=np.uint8)
    DataStructs.cDataStructs.ConvertToNumpyArray(vect, bits)
    return bits

def bfp_from_bytes(fp_bytes, size=None):
    bits = np.unpackbits(fp_bytes)
    actual_size = len(bits)
    if size is not None and actual_size != size:
        if actual_size % size == 0:
            fold_factor = actual_size // size
        else:
            raise ValueError("BFP size {0} does not match expected {1}".format(len(bits), size))
    else:
        fold_factor = None
    vect = bfp_from_bits(bits, size=actual_size)
    if fold_factor:
        vect = DataStructs.FoldFingerprint(vect, fold_factor)
    return vect

def bfp_to_bytes(vect):
    bits = bfp_to_bits(vect)
    packed = np.packbits(bits)
    return packed

def bfp_from_chars(chars, size=None):
    byte_values = bytes_from_chars(chars)
    vect = bfp_from_bytes(byte_values, size=size)
    return vect
    
def bfp_to_chars(vect):
    byte_values = bfp_to_bytes(vect)
    chars = bytes_to_chars(byte_values)
    return chars

def bfp_from_binary_text(binary_text, size=None):
    byte_values = bytes_from_binary_text(binary_text)
    vect = bfp_from_bytes(byte_values, size=size)
    return vect

def bfp_to_binary_text(vect):
    byte_values = bfp_to_bytes(vect)
    binary_text = bytes_to_binary_text(byte_values)
    return binary_text
    
def bfp_from_base64(data, size=None, altchars='+/'):
    data = str(data)
    raw = base64.b64decode(data, altchars)
    array = np.frombuffer(raw, dtype=np.uint8)
    vect = bfp_from_bytes(array, size=size)
    return vect

def bfp_to_base64(vect, altchars='+/'):
    array = bfp_to_bytes(vect)
    raw = array.data
    encoded = base64.b64encode(raw, altchars)
    return encoded

def bfp_from_base64rdk(data, size=None):
    if size is None:
        raise ValueError("Cannot create RDKit base64 fingerprint without size")
    bfp = DataStructs.ExplicitBitVect(size)
    try:
        bfp.FromBase64(data)
    except IndexError:
        bfp.FromBase64(data.replace(' ', '+'))
    return bfp

def bfp_to_base64rdk(vect):
    return vect.ToBase64()
    
def bfp_from_base64fp(data, size=None):
    return bfp_from_base64(data, size=size, altchars='.+')

def bfp_to_base64fp(vect):
    return bfp_to_base64(vect, '.+')

BFP_PARSERS = [
    ('bfp', ensure_bfp),
    ('element', extract_bfp_element),
    ('bytes', bfp_from_bytes),
    ('chars', bfp_from_chars),
    ('binary_text', bfp_from_binary_text),
    ('bits', bfp_from_bits),
    ('binary', bfp_from_raw_binary_text),
    ('base64rdk', bfp_from_base64rdk),
    ('base64fp', bfp_from_base64fp),
    ('base64', bfp_from_base64),
    # TODO: base64, etc.
]

def attempt_bfp_conversion(data, size=None, method=None, raw_method=None):
    errors = []

    # Special case for mol elements (and convertable to mol)
    mol = None
    if isinstance(data, DataStructs.ExplicitBitVect):
        pass
    elif isinstance(data, Chem.Mol):
        mol = data
    elif hasattr(data, 'as_mol'):
        mol = data.as_mol
    elif hasattr(data, 'mol'):
        mol = data.mol
    elif raw_method is not None and isinstance(data, basestring) and '\x00' not in data:
        try:
            mol = raw_method(data)
        except (ValueError, TypeError, AttributeError) as error:
            errors.append(str(error))

    if mol is not None:
        if method is not None:
            data = method(mol)
        else:
            raise ValueError("Attempting to generate bfp from Mol "
                             "but no method provided")
    elif isinstance(data, (basestring, buffer)):
        data = str(data)

    
    # Try all known bfp parsers
    for fmt, parser in BFP_PARSERS:
        try:
            bfp = parser(data, size=size)
            return fmt, bfp
        except (ValueError, TypeError) as error:
            errors.append(str(error))

    raise ValueError("Failed to convert `{0}` to bfp. Errors were: {1}".format(data, ", ".join(errors)))
    
def coerce_to_bfp(data, size=None, method=None, raw_method=None):
    fmt, bfp = attempt_bfp_conversion(data, size=size, method=method, raw_method=raw_method)
    return bfp


## Core Types #######################################################
## TODO: Some of this turned out to be unused. Clean up
## Functions need a lot of work. Still need to figure that one out
## Consult geoalchemy2 and razi for possible solutions


class _RDKitFunction(functions.GenericFunction):
    HELP = 'See http://www.rdkit.org/docs/Cartridge.html'
    
    def __init__(self, *args, **kwargs):
        try:
            expr = kwargs.pop('expr')
            args = (expr,) + args
        except KeyError:
            pass
        super(_RDKitFunction, self).__init__(*args, **kwargs)


def _hybrid_function(sql_function, local_function, 
                     type_=None, 
                     as_property=True,
                     local_in_type=None,
                     args_in_types=None,
                     local_kwargs=None,
                     element_kwargs=None,
                     sql_cast_out=None,
                     help="", tpl=functions.GenericFunction,
                     register=True):

    if local_kwargs is None:
        local_kwargs = {}
    if element_kwargs is None:
        element_kwargs = {}

    attributes = {
        'name': sql_function,
        '_as_property': as_property,
        '_sql_cast_out': sql_cast_out,
        '_local_in_type': local_in_type,
        '_args_in_types': args_in_types,
        '_local_kwargs': local_kwargs,
        '_element_kwargs': element_kwargs,
    }

    docs = [help]
    docs.append(getattr(tpl, 'HELP', ''))

    if local_function is not None:
        attributes['local_function'] = staticmethod(local_function)

    if type_ is not None:
        element = getattr(type_, 'element', None)
        type_str = ".".join((type_.__module__, type_.__name__))
        docs.append("Return Type: {type}".format(type=type_str))
        attributes['type'] = type_
        attributes['_element'] = element

    attributes['__doc__'] = "\n\n".join(docs)

    cls_name = sql_function
    cls = type(cls_name, (tpl,), attributes)

    if register:
        globals()[cls_name] = cls

    return cls


def _rdkit_function(sql_function, local_function, 
                    type_=None, as_property=True,
                    local_in_type=None, sql_cast_out=None,
                    args_in_types=None,
                    help="", register=True):
    return _hybrid_function(sql_function, local_function, 
                            as_property=as_property, 
                            type_=type_, 
                            local_in_type=local_in_type, 
                            sql_cast_out=sql_cast_out,
                            args_in_types=args_in_types,
                            tpl=_RDKitFunction,
                            help=help, register=register)


@compiles(_RDKitFunction)
def _compile_rdkit_function(element, compiler, **kwargs):
    if element._args_in_types:
        new_args = []
        clauses = list(element.clauses)
        for idx, (clause, type_) in enumerate(zip(clauses, element._args_in_types)):
            if type_ is None:
                new_args.append(clause)
            else:
                if isinstance(type_, int):
                    tpl = clauses[type_]
                    if hasattr(tpl, 'type'):
                        type_ = tpl.type
                    elif hasattr(tpl, 'type_'):
                        type = tpl.type_
                    else:
                        raise ValueError("Couldn't find 'type' from argument {} ({})".format(type_, tpl))
                expr = type_.bind_expression(clause.effective_value)
                if hasattr(expr, 'desc'):
                    expr = expr.desc
                name = '_{}_dyn_{}_{}'.format(element.name, type_.name, idx)
                expr = expression.bindparam(name, expr)
                new_args.append(expr)
        clauses = elements.ClauseList(*new_args)
        compiled = "{0}({1})".format(element.name, compiler.process(clauses))
    else:
        compiled = compiler.visit_function(element)
    if element._sql_cast_out is not None:
        compiled = "CAST({call} AS {cast})".format(call=compiled, 
                                           cast=element._sql_cast_out)
    return compiled



class _RDKitFunctionCallable(object):

    @property
    def _functions(self):
        return getattr(self, '_function_namespace', functions._FunctionGenerator)

    def _get_function(self, fn_name):
        fn = getattr(self._functions, fn_name)
        return fn

    def _can_call_function_locally(self, fn_name):
        return hasattr(self._get_function(fn_name), 'local_function')\
               and getattr(self, '_allow_local_functions', True)

    def _function_call(self, fn_name, data=None):
        if data is None:
            data = self
        fn = self._get_function(fn_name)
        if getattr(fn, '_as_property', True):
            return fn(expr=data)
        else:
            def partial(*args):
                return fn(expr=data, *args)
            return partial


def instrumented_property(name):
    def fn(self, *args, **kwargs):
        return self._instrumented_target(name, *args, **kwargs)
    return property(fn)

    
class _RDKitInstrumentedFunctions(object):
    def _instrumented_target(self, name, data=None, *args, **kwargs):
        return self._function_call(name, data)


class RDKitInstrumentedColumnClass(object):
    @property
    def __object_data_column__(self):
        raise NotImplemented("_target column not set for {:r}".format(self))

    def _instrumented_target(self, name, *args, **kwargs):
        target = getattr(self, self.__object_data_column__)
        value = getattr(target, name)
        return value


class _RDKitElement(object):

    def __str__(self):
        return str(self.desc)
    def __repr__(self):
        return "<{0} at {1}>".format(self.__class__.__name__, id(self))


class _RDKitDataElement(_RDKitElement, _RDKitFunctionCallable, _RDKitInstrumentedFunctions):
    """ Base datatype for object that need explicit modification
        either into or out of psql.
        Define "compile_desc_literal" to convert
    """
    def __init__(self, data):
        self.data = data

    def _function_call(self, fn_name, data=None):
        if data is not None:
            raise ValueError("Cannot call bound RDKit function with additional data")

        if isinstance(self.data, (expression.BindParameter, 
                                  expression.ColumnElement))\
           or not self._can_call_function_locally(fn_name):
            return super(_RDKitDataElement, self)\
                        ._function_call(fn_name, data=self)
        else:
            fn = self._get_function(fn_name)
            local_fn = fn.local_function
            local_data = self.to_local_type()
            element = getattr(fn, '_element', None)
            if element is None:
                element = lambda x, **_: x

            treat_as_property = getattr(fn, '_as_property', True)
            local_kwargs = getattr(fn, '_local_kwargs', {})
            element_kwargs = getattr(fn, '_element_kwargs', {})

            def partial_fn(*args):
                raw = local_fn(local_data, *args, **local_kwargs)
                result = element(raw, **element_kwargs)
                return result

            if treat_as_property:
                return partial_fn()
            else:
                return partial_fn

    def __getattr__(self, name):
        if name == 'data':
            raise AttributeError("Data not defined")
        return self._function_call(name)

    @property
    def desc(self):
        if isinstance(self.data, expression.BindParameter):
            return self.data
        else:
            return self.compile_desc_literal()

    def _compiler_dispatch(self, compiler, **kwargs):
        if hasattr(self.desc, '_compiler_dispatch'):
            return self.desc._compiler_dispatch(compiler, **kwargs)
        else:
            param = bindparam(key=None, value=self.desc, type_=self.type)
            return compiler.visit_bindparam(param, **kwargs)
        
    def compile_desc_literal(self):
        raise NotImplemented

    def to_local_type(self):
        raise NotImplemented

    @property
    def local(self):
        raise NotImplementedError('')

    @property
    def bind(self):
        bound_element = self.new_with_attrs(self.data, local_functions=False)
        return bound_element 

    def new_with_attrs(self, data, local_functions=True):
        cls = type(self)
        params = self._get_params()
        new = cls(data, **params)
        new._allow_local_functions = local_functions
        return new

    def _get_params(self):
        return {}


## Mol element interface.
## Either implicit (punt to postgres) or explicit (define functions
## to perfrom in/out conversions)


class RDKitMolProperties(object):
    @property
    def _function_namespace(self):
        return _RDKitMolFunctions
    
    mwt = instrumented_property('mwt')
    logp = instrumented_property('logp')
    tpsa = instrumented_property('tpsa') 
    hba = instrumented_property('hba') 
    hbd = instrumented_property('hbd') 
    num_atoms = instrumented_property('num_atoms')
    num_heavy_atoms = instrumented_property('num_heavy_atoms')
    num_rotatable_bonds = instrumented_property('num_rotatable_bonds')
    num_hetero_atoms = instrumented_property('num_hetero_atoms')
    num_rings = instrumented_property('num_rings')
    num_aromatic_rings = instrumented_property('num_aromatic_rings')
    num_aliphatic_rings = instrumented_property('num_aliphatic_rings')
    fractioncsp3 = instrumented_property('fractioncsp3')
    inchi = instrumented_property('inchi')
    inchikey = instrumented_property('inchikey')

    rdkit_fp = instrumented_property('rdkit_fp')
    morgan_fp = instrumented_property('morgan_fp')


class RDKitBfpProperties(object):
    @property
    def _function_namespace(self):
        return _RDKitBfpFunctions
    
    size = instrumented_property('size')

    tanimoto = instrumented_property('tanimoto')
    dice = instrumented_property('dice')


class RawMolElement(_RDKitDataElement, RDKitMolProperties):
    """ Base mol element. Let postgres deal with it. 
        Also define explicit conversion methods for mols"""

    def __init__(self, mol, _force_sanitized=True):
        _RDKitDataElement.__init__(self, mol)
        self._mol = None
        self.force_sanitized = _force_sanitized

    def _get_params(self):
        return {
            '_force_sanitized': self.force_sanitized,
        }

    def compile_desc_literal(self):
        return self.mol_cast(self.data)

    def to_local_type(self):
        return self.as_mol

    def embed_mol_2d(self):
        mol = self.as_mol
        if mol.GetNumConformers() == 0:
            mol.Compute2DCoords()
            return True
        else:
            return False

    def embed_mol_3d(self):
        mol = self.as_mol
        if mol.GetNumConformers() == 0 or not mol.GetConformer().Is3D():
            EmbedMolecule(mol)
            return True
        else:
            return False
    
    @staticmethod
    def mol_cast(value, qmol=False):
        if qmol:
           type_ = QMol
        else:
           type_ = Mol
        return expression.cast(value, type_)
        
    @property
    def as_mol(self):
        # Try and convert to rdkit.Chem.Mol (if not already a Chem.Mol)
        # Cache the mol to persist props and prevent recomputation
        if self._mol is None:
            self._mol = coerce_to_mol(self.data, sanitize=self.force_sanitized)
        return self._mol

    @property
    def as_mol2d(self):
        self.embed_mol_2d()
        return self.as_mol

    @property
    def as_mol3d(self):
        self.embed_mol_3d()
        return self.as_mol
    
    @property
    def as_binary(self): 
        return buffer(self.as_mol.ToBinary())
    
    @property
    def as_smiles(self):
        return Chem.MolToSmiles(self.as_mol, isomericSmiles=True)

    @property
    def as_flat_smiles(self):
        return Chem.MolToSmiles(self.as_mol, isomericSmiles=False)
    
    @property
    def as_smarts(self):
        return Chem.MolToSmarts(self.as_mol, isomericSmiles=True)
    
    @property
    def as_ctab(self):
        return Chem.MolToMolBlock(self.as_mol, includeStereo=True)
    
    @property
    def as_sdf(self):
        return self.as_ctab
    
    @property
    def as_pdb(self):
        return Chem.MolToPDBBlock(self.as_mol)
    
    @property
    def as_inchi(self):
        return Chem.MolToInchi(self.as_mol)
    
    @property
    def as_inchikey(self):
        return Chem.InchiToInchiKey(self.as_inchi)

    def get_image(self, *args, **options):
        return MolToImage(self.as_mol, *args, **options)
    

class _ExplicitMolElement(RawMolElement, expression.Function):
    """ Define a mol that expects a specific IO format """

    def __init__(self, mol, _force_sanitized=True):        
        RawMolElement.__init__(self, mol, _force_sanitized)
        expression.Function.__init__(self, self.backend_in, self.desc,
                                           type_=Mol(coerce_=self.frontend_coerce))
        
    @property
    def backend_in(self): 
        raise NotImplementedError
        
    @property
    def backend_out(self): 
        raise NotImplemented
    
    @property
    def frontend_coerce(self): 
        raise NotImplemented
        

class BinaryMolElement(_ExplicitMolElement):   
    backend_in = 'mol_from_pkl'
    backend_out = 'mol_to_pkl'
    frontend_coerce = 'binary'

    def __init__(self, mol, _force_sanitized=True):
        if isinstance(mol, basestring):
            mol = bytes(mol)
        super(BinaryMolElement, self).__init__(mol, _force_sanitized=_force_sanitized)

    def compile_desc_literal(self):
        return self.as_binary

    def __str__(self):
        return "<Binary {}>".format(self.as_smiles)

    def __repr__(self):
        return "<{0}; <{2}>>".format('BinaryMolElement', id(self), self.as_smiles)

    def __getstate__(self):
        d = dict(self.__dict__)
        d['data'] = str(d['data'])
        del d['clause_expr']
        return d

    def __setstate__(self, d):
        data = d['data']
        BinaryMolElement.__init__(self, data, d['force_sanitized'])

    

class SmartsMolElement(RawMolElement):
    backend_in = 'mol_from_smarts'
    backend_out = 'mol_to_smarts'
    frontend_coerce = 'smarts'

    @property
    def as_mol(self):
        # Try and convert to rdkit.Chem.Mol (if not already a Chem.Mol)
        # Cache the mol to persist props and prevent recomputation
        if self._mol is None:
            try:
                self._mol = smarts_to_mol(self.data, sanitize=self.force_sanitized)
            except ValueError:
                self._mol = coerce_to_mol(self.data, sanitize=self.force_sanitized)
        return self._mol

    @property
    def _force_sql_type(self):
        return QMol

    def compile_desc_literal(self):
        return self.mol_cast(self.as_smarts, qmol=True)
    
    
class SmilesMolElement(_ExplicitMolElement):   
    backend_in = 'mol_from_smiles'
    backend_out = 'mol_to_smiles'
    frontend_coerce = 'smiles'
    
    def compile_desc_literal(self):
        return self.as_smiles
    
class CtabMolElement(_ExplicitMolElement):
    backend_in = 'mol_from_ctab'
    backend_out = 'mol_to_ctab'
    frontend_coerce = 'ctab'
    
    def compile_desc_literal(self):
        return self.as_ctab


## Binary fingerprint elements
## Handle conversion from various formats (some RDKit based and others)


class BfpElement(_RDKitDataElement, expression.Function, RDKitBfpProperties):
    
    backend_in = 'bfp_from_binary_text'
    backend_out = 'bfp_to_binary_text'

    def __init__(self, fp, size=None, method=None, raw_method=None):
        _RDKitDataElement.__init__(self, fp)
        self._size = size
        self._method = method
        self._raw_method = raw_method
        self._fp = None
        expression.Function.__init__(self, self.backend_in, self.desc,
                                           type_=Bfp(bits=self._size,
                                                     method=self._method,
                                                     raw_method=self._raw_method))

    def _get_params(self):
        return {
            'size': self._size,
            'method': self._method,
            'raw_method': self._raw_method,
        }

    def compile_desc_literal(self):
        return self.as_binary_text

    def to_local_type(self):
        return self
    
    @property
    def as_bfp(self):
        if self._fp is None:
            self._fp = coerce_to_bfp(self.data, 
                                     size=self._size,
                                     method=self._method,
                                     raw_method=self._raw_method)
        return self._fp
    
    @property
    def as_binary(self):
        return bfp_to_binary(self.as_bfp)
    
    @property
    def as_bytes(self):
        return bfp_to_bytes(self.as_bfp)
    
    @property
    def as_chars(self):
        return bfp_to_chars(self.as_bfp)
    
    @property
    def as_bits(self):
        return bfp_to_bits(self.as_bfp)
    
    @property
    def as_binary_text(self):
        return bfp_to_binary_text(self.as_bfp)
        
    @property
    def as_base64(self):
        return bfp_to_base64(self.as_bfp)
        
    @property
    def as_base64fp(self):
        return bfp_to_base64fp(self.as_bfp)
    
    @property
    def as_array(self):
        return self.as_bytes


class _RDKitComparator(UserDefinedType.Comparator, _RDKitFunctionCallable):
    key = None
    _element = None

    def _function_call(self, name, data=None):
        if data is not None:
            raise ValueError("Cannot call bound RDKit function with additional data")
        return super(_RDKitComparator, self)\
                    ._function_call(name, data=self.expr)

    def _should_cast(self, obj):
        if self._element is None:
            return False
        elif isinstance(obj, self._element):
            return False
        elif isinstance(obj, expression.BindParameter):
            return False
        elif getattr(obj, '_element', None) == self._element:
             return False
        else:
            return True

    def _cast_other_element(self, obj):
        raise NotImplemented

    @staticmethod
    def _ensure_other_element(fn):
        @functools.wraps(fn)
        def wrapper(self, other):
            other = self._cast_other_element(other)
            bound_clause = fn(self, other)
            return bound_clause
        return wrapper


class _RDKitMolComparator(_RDKitComparator, 
                          _RDKitInstrumentedFunctions, 
                          RDKitMolProperties):
    _element = RawMolElement
    force_sanitize = False

    COERSIONS = {
        'smiles': SmilesMolElement,
        'smarts': SmartsMolElement,
        'binary': BinaryMolElement,
        'ctab': CtabMolElement,

        # Special Case Explicit Mol Elements
        'mol': BinaryMolElement,
    }

    @property
    def _sanitize(self):
        try:
            return self.sanitized
        except AttributeError:
            return self.type.sanitized

    def _cast_other_element(self, obj, cast_as=RawMolElement):
        if self._should_cast(obj):
            fmt = infer_mol_format(obj, sanitize=self._sanitize)
            convert = self.COERSIONS.get(fmt, cast_as)
            other = convert(obj, _force_sanitized=self._sanitize)
        else:
            other = obj
        return other

    #@_RDKitComparator._ensure_other_element
    #def structure_is(self, other):
    #    return self.operate(operators.eq, other)

    @_RDKitComparator._ensure_other_element
    def structure_is(self, other):
        return self.op('@=', is_comparison=True)(other)

    @_RDKitComparator._ensure_other_element
    def has_substructure(self, other):
        return self.op('@>', is_comparison=True)(other)

    @_RDKitComparator._ensure_other_element
    def in_superstructure(self, other):
        return self.op('<@', is_comparison=True)(other)

    def contains(self, other, escape=None):
        return self.has_substructure(other)

    def contained_in(self, other):
        return self.in_superstructure(other)

    def match(self, other, escape=None):
        other = SmartsMolElement(other, _force_sanitized=self._sanitize)
        return self.has_substructure(other)

    def matched_by(self, other, escape=None):
        other = SmartsMolElement(other, _force_sanitized=self._sanitize)
        return self.in_superstructure(other)

    def __eq__(self, other):
        expr = self.structure_is(other)
        expr = CustomEqualityBinaryExpression_HACK._override_expr(expr, ('@=',))
        return expr

    def __ne__(self, other):
        return ~self.__eq__(other)

    def __le__(self, other):
        return self.has_substructure(other)

    def __ge__(self, other):
        return self.has_superstructure(other)

    def __contains__(self, other):
        return self.has_substructure(other)
        
    def __getitem___(self, other):
        return self.has_substructure(other)


## SQLAlchemy data types


class Mol(UserDefinedType):
    name = 'mol'
    default_element = RawMolElement
    base_element = RawMolElement
    comparator_factory = _RDKitMolComparator

    def __init__(self, coerce_='smiles', sanitized_=True):
        self.coerce = coerce_
        self.sanitized = sanitized_

    def _coerce_compared_value(self, op, value):
        return self
    
    def _get_coerced_element(self, default=None):
        return self.comparator_factory.COERSIONS.get(self.coerce, default)
    
    def get_col_spec(self):
        return self.name
    
    def bind_expression(self, bindvalue):
        element = None
        if isinstance(bindvalue, expression.BindParameter):
            effective_value = bindvalue.effective_value
            if isinstance(effective_value, RawMolElement):
                value = bindvalue.effective_value.desc
                if getattr(value, 'is_clause_element', False):
                    element = value
                else:
                    bindvalue = expression.BindParameter(key=None, value=value, type_=Mol)
        if element is None:
            sanitize = self.sanitized
            element_type = self._get_coerced_element(default=self.default_element)
            element = element_type(bindvalue, _force_sanitized=sanitize)
        return element
    
    def column_expression(self, col):
        element = self._get_coerced_element()
        if element is not None:
            fn_name = element.backend_out
            fn = getattr(expression.func, fn_name)
            return fn(col, type_=self)
        else:
            return self.default_element.mol_cast(col)
    
    def bind_processor(self, dialect):
        def process(value):
            if isinstance(value, self.default_element):
                return value.desc
            else:
                return value  # This may need to do further coersion
        return process
    
    def result_processor(self, dialect, coltype):
        element = self._get_coerced_element(default=self.default_element)
        def process(value):
            if value is not None:
                return element(value, _force_sanitized=self.sanitized)
            else:
                return None
        return process


class QMol(Mol):
    name = 'qmol'
    default_element = SmartsMolElement

    def __init__(self, coerce_='smarts', sanitized_=False):
        super(QMol, self).__init__(coerce_=coerce_, 
                                        sanitized_=sanitized_)


class BinaryMol(Mol):
    default_element = BinaryMolElement

    def __init__(self, coerce_='binary', sanitized_=True):
        super(BinaryMol, self).__init__(coerce_=coerce_, 
                                        sanitized_=sanitized_)


class _RDKitBfpComparator(_RDKitComparator,
                          _RDKitInstrumentedFunctions,
                          RDKitBfpProperties):
    _element = BfpElement

    def _cast_other_element(self, obj):
        if self._should_cast(obj):
            try:
                size = self.type.bits
            except AttributeError:
                try:
                    size = getattr(self, '_size')
                except AttributeError:
                    size = None
            try:
                method = self.type.method
            except AttributeError:
                try:
                    method = getattr(self, '_method')
                except AttributeError:
                    method = None
            try:
                raw_method = self.type.raw_method
            except AttributeError:
                try:
                    raw_method = getattr(self, '_raw_method')
                except AttributeError:
                    raw_method = None
            other = coerce_to_bfp(obj, size=size, method=method, raw_method=raw_method)
            element = self._element(other)
        else:
            element = obj
        return element

    @_RDKitComparator._ensure_other_element
    def tanimoto_similar(self, other):
        return self.op('%%', is_comparison=True)(other)
    
    @_RDKitComparator._ensure_other_element
    def dice_similar(self, other):
        return self.op('#', is_comparison=True)(other)
    
    @_RDKitComparator._ensure_other_element
    def tanimoto_nearest_neighbors(self, other):
        ordering = self.op('<%%>')(other)
        ordering.type = DOUBLE_PRECISION()
        return ordering

    @_RDKitComparator._ensure_other_element
    def dice_nearest_neighbors(self, other):
        ordering = self.op('<#>')(other)
        ordering.type = DOUBLE_PRECISION()
        return ordering

    @_RDKitComparator._ensure_other_element
    def __eq__(self, other):
        return super(_RDKitBfpComparator, self).__eq__(other)


class Bfp(UserDefinedType):
    name = 'bfp'
    element = BfpElement
    bits = None
    method = None
    raw_method = None

    comparator_factory = _RDKitBfpComparator
    
    def __init__(self, bits=None, method=None, raw_method=None):
        self.bits = bits or getattr(self, 'bits', None)
        self.method = method or getattr(self, 'method', None)
        self.raw_method = raw_method or getattr(self, 'raw_method', None)
    
    def get_col_spec(self):
        return self.name
    
    def bind_expression(self, bindvalue):
        element = self.element(bindvalue, 
                               size=self.bits, 
                               method=self.method,
                               raw_method=self.raw_method)
        return element
    
    def column_expression(self, col):
        fn_name = self.element.backend_out
        fn = getattr(expression.func, fn_name)
        return fn(col, type_=self)
    
    def bind_processor(self, dialect):
        def process(value):
            if isinstance(value, self.element):
                return value.desc
            else:
                return value  # This may need to do further coersion
        return process
    
    def result_processor(self, dialect, coltype):
        def process(value):
            if value is not None:
                return self.element(value, size=self.bits,
                                           method=self.method,
                                           raw_method=self.raw_method)
            else:
                return None
        return process

    def __getattr__(self, name):
        return getattr(functions, name)(self)

## TODO: Fix these to be accessible as attributes of Mol/BFP objects
##       as well as elements. Will need to also define the 'local' 
##       equivalents of each function to make mol weight computable
##       in the same way without touching the database.


class _RDKitMolFunctions(object):
    # Descriptors
    mwt = _rdkit_function(
                'mol_amw', 
                Descriptors.MolWt,
                type_=Float,
                help="Returns the AMW for a molecule.")
    logp = _rdkit_function(
                'mol_logp', 
                Descriptors.MolLogP,
                type_=Float,
                help="Returns the LogP for a molecule.")
    tpsa = _rdkit_function(
                'mol_tpsa', 
                Descriptors.TPSA,
                type_=Float,
                help="Returns the topological polar surface  area for a "
                     "molecule.")
    hba = _rdkit_function(
                'mol_hba', 
                Descriptors.NumHAcceptors,
                type_=Integer,
                help="Returns the number of Lipinski H-bond acceptors for a "
                     "molecule")
    hbd = _rdkit_function(
                'mol_hbd', 
                Descriptors.NumHDonors,
                type_=Integer,
                help="Returns the number of Lipinski H-bond donors for a "
                     "molecule")
    num_atoms = _rdkit_function(
                'mol_numatoms', 
                Chem.Mol.GetNumAtoms,
                type_=Integer,
                help="Returns the number of atoms in a molecule")
    num_heavy_atoms = _rdkit_function(
                'mol_numheavyatoms', 
                Chem.Mol.GetNumHeavyAtoms,
                type_=Integer,
                help="Returns the number of heavy atoms in a molecule")
    num_rotatable_bonds = _rdkit_function(
                'mol_numrotatablebonds', 
                Descriptors.NumRotatableBonds,
                type_=Integer,
                help="Returns the number of rotatable bonds in a molecule")
    num_hetero_atoms = _rdkit_function(
                'mol_numheteroatoms', 
                Descriptors.NumHeteroatoms,
                type_=Integer,
                help="Returns the number of heteroatoms in a molecule")
    num_rings = _rdkit_function(
                'mol_numrings', 
                Descriptors.RingCount,
                type_=Integer,
                help="Returns the number of rings in a molecule")
    num_aromatic_rings = _rdkit_function(
                'mol_numaromaticrings',
                Descriptors.NumAromaticRings,
                type_=Integer,
                help="Returns the number of aromatic rings in a molecule")
    num_aliphatic_rings = _rdkit_function(
                'mol_numaliphaticrings',
                Descriptors.NumAliphaticRings,
                type_=Integer,
                help="Returns the number of aliphatic rings in a molecule")
    num_saturated_rings = _rdkit_function(
                'mol_numsaturatedrings', 
                Descriptors.NumSaturatedRings,
                type_=Integer,
                help="Returns the number of saturated rings in a molecule")
    num_aromaticheterocycles = _rdkit_function(
                'mol_numaromaticheterocycles',
                Descriptors.NumAromaticHeterocycles,
                type_=Integer,
                help="Returns the number of aromatic heterocycles in a molecule")
    formula = _rdkit_function(
                'mol_formula', 
                Chem.rdMolDescriptors.CalcMolFormula,
                type_=String,
                help='Returns a string with the molecular formula. The second '
                     'argument controls whether isotope information is '
                     'included in the formula; the third argument controls '
                     'whether "D" and "T" are used instead of [2H] and [3H].')

    fractioncsp3 = _rdkit_function(
                'mol_fractioncsp3',
                Chem.rdMolDescriptors.CalcFractionCSP3,
                type_=Float,
                help="Returns the fraction of C atoms that are SP3 hybridized")

    to_pkl = _rdkit_function(
                'mol_to_pkl',
                Chem.Mol.ToBinary,
                type_=BINARY,
                sql_cast_out='bytea')

    inchi = _rdkit_function(
                'mol_inchi', 
                Chem.MolToInchi,
                type_=Text,
                sql_cast_out='text',
                help='Returns an InChI for the molecule. (available '
                     'from the 2011_06 release, requires that the RDKit be '
                     'uilt with InChI support).')

    inchikey = _rdkit_function(
                'mol_inchikey', 
                lambda m: Chem.InchiToInchiKey(Chem.MolToInchi(m)),
                type_=String,
                sql_cast_out='bpchar',
                help='Returns an InChI key for the molecule. (available '
                     'from the 2011_06 release, requires that the RDKit be '
                     'uilt with InChI support).')

    rdkit_fp = _rdkit_function(
                'rdkit_fp',
                Chem.RDKFingerprint,
                as_property=True,
                type_=Bfp)
            
    morgan_fp = _rdkit_function(
                'morganbv_fp',
                Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect,
                as_property=False,  # Need to supply a radius
                type_=Bfp)


def _rdkit_bfp_uniary_fn(fn):
    def wrapped(a, *args, **kwargs):
        a_bfp = coerce_to_bfp(a)
        return fn(a, *args, **kwargs)
    return wrapped

def _rdkit_bfp_binary_fn(fn):
    def wrapped(a, b, *args, **kwargs):
        a_bfp = coerce_to_bfp(a)
        b_bfp = coerce_to_bfp(b)
        return fn(a_bfp, b_bfp, *args, **kwargs)
    return wrapped


class _RDKitBfpFunctions(object):
    size = _rdkit_function(
                'size',
                _rdkit_bfp_uniary_fn(DataStructs.ExplicitBitVect.GetNumBits),
                type_=Integer,
                help="Returns the number of bits in a binary fingerprint")

    tanimoto = _rdkit_function(
                'tanimoto_sml',
                _rdkit_bfp_binary_fn(DataStructs.TanimotoSimilarity),
                args_in_types=(None, 0),
                sql_cast_out='numeric',
                as_property=False)

    dice = _rdkit_function(
                'dice_sml',
                _rdkit_bfp_binary_fn(DataStructs.DiceSimilarity), 
                args_in_types=(None, 0),
                sql_cast_out='numeric',
                as_property=False)


## Code to handle modifying the similarity search threshold constants 
## Borrowed straight from RAZI


class GUC(expression.Executable, expression.ClauseElement):
    """ From Razi """

    def __init__(self, variable, type_=lambda x: x, 
                       default=None, 
                       *args, **kwargs):
        self.variable = variable
        self.type_ = type_
        if default is not None:
            self.set(default)

    def set_in_session(self, value):
        def transform(query):
            query.session.execute(self.set_expression(value))
            return query
        return transform

    def set_expression(self, value):
        value = self.type_(value)
        query = 'SET {variable} TO :value'.format(variable=self.variable)
        expr = expression.text(query)
        preped_expr = expr.execution_options(autocommit=False)\
                          .params(value=value)
        return preped_expr

    def setter(self, value):
        return lambda engine: engine.execute(self.set_expression(value))

    def set(self, engine, value):
        return self.setter(value)(engine)

    def get_expression(self):
        query = 'SHOW {variable}'.format(variable=self.variable)
        expr = expression.text(query)
        return expr

    def getter(self):
        return lambda engine: self.type_(engine.scalar(self.get_expression()))

    def get(self, engine):
        return self.getter()(engine)

    @contextlib.contextmanager
    def set_in_context(self, engine, value):
        original = self.get(engine)
        self.set(engine, value)
        yield
        self.set(engine, original)

    def __call__(self, engine, value):
        return self.set_in_context(engine, value)


@compiles(GUC)
def __compile_guc(element, compiler, **kwargs):
    return compiler.process(element.get())


tanimoto_threshold = GUC('rdkit.tanimoto_threshold', float)
dice_threshold = GUC('rdkit.dice_threshold', float)


class RDKitMolClass(RDKitInstrumentedColumnClass, RDKitMolProperties):
    pass


class RDKitBfpClass(RDKitInstrumentedColumnClass, RDKitBfpProperties):
    pass


def convert_to(value, type_, **kwargs):
    if hasattr(type_, 'type'):
        type_ = type_.type
    #elif hasattr(type_, 'python_type'):
    #    type_ = type_.type
    if hasattr(value, 'force_type'):
        converter = value.force_type.bind_expression
    else:
        converter = type_.bind_expression
    return converter(value)


def converter_to(type_, **kwargs):
    def converter(data):
        return convert_to(data, type_, **kwargs)
    return converter


## Define the rdkit datatypes in sqlalchemy

ischema_names['mol'] = Mol
ischema_names['qmol'] = QMol
ischema_names['bfp'] = Bfp

#_RDKit_Mol_Functions._inject(RawMolElement)
#_RDKit_Mol_Functions._inject(Mol.comparator_factory)

#######################################################################
## Test Case
#######################################################################
if __name__ == '__main__':
    from sqlalchemy import (create_engine, MetaData, Column, Integer, String,
            func, select)
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.ext.declarative import declarative_base
    
    engine = create_engine('postgresql+psycopg2://zincread:@zincdb-1/zinc', echo=True)
    metadata = MetaData(engine)
    Base = declarative_base(metadata=metadata)
    
    db = sessionmaker(bind=engine)()
    
    class Substance(Base):
        __tablename__ = 'substance'
        
        sub_id = Column('sub_id', Integer, primary_key=True)
        structure = Column('smiles', Mol)
        num_heavy_atoms = Column('n_hvy_atoms', Integer)
        name = Column('name', String)
        
        @property
        def mol(self):
            return self.structure.as_mol

        @property
        def smiles(self):
            return self.structure.as_smiles
        
        @property
        def smarts(self):
            return self.structure.as_smarts
        
        @property
        def inchi(self):
            return self.structure.as_inchi
        
        @property
        def inchikey(self):
            return self.structure.as_inchikey
    
    
    aspirin = db.query(Substance).filter(Substance.sub_id==53).one()
    print aspirin.smiles
    print aspirin.structure.as_pdb
    
    tenbenz = db.query(Substance)\
                .filter(Substance.structure.contains('c1cccccc1'))\
                .limit(10)
    
    for substance in tenbenz:
        print substance.sub_id, substance.smarts
