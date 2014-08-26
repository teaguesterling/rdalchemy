#!/usr/bin/env python
#
# Copyright (C) 2014 Teague Sterling, John Irwin,
# Regents of the University of California
#
# This implementation utilizes code and methods from Riccardo Vianello
# as well as code structure and inspiration from geoalchemy2

import contextlib
import functools
import operator
import types

import numpy as np

from sqlalchemy import event, Table
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.sql import expression, functions, type_coerce
from sqlalchemy.types import UserDefinedType, _Binary, TypeDecorator
from sqlalchemy.dialects.postgresql.base import ischema_names

from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors


## Datatype Converstions
## TODO: This should be reorganized... do we even want 
##       automatic conversion from ctab to mol?
## Break these out into separate files for each type


## Mol conversions ##################################################

def ensure_mol(mol, sanitize=True):
    if not isinstance(mol, Chem.Mol):
        raise ValueError("Not already an instance of rdkit.Chem.Mol")
    if sanitize:
        Chem.SanitizeMol(mol)
    return mol

def extract_mol_element(mol, sanitize=True):
    if isinstance(mol, RawMolElement):
        mol = mol.as_mol
    else:
        raise ValueError("Not an instance of RawMolElement")
    if sanitize:
        Chem.SanitizeMol(mol)
    return mol

def smiles_to_mol(smiles, sanitize=True):
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        raise ValueError("Failed to parse SMILES: `{0}`".format(smiles))
    if sanitize:
        Chem.SanitizeMol(mol)
    return mol

def smarts_to_mol(smarts, sanitize=True):
    mol = Chem.MolFromSmarts(smarts)
    if mol is None:
        raise ValueError("Failed to parse SMARTS: `{0}`".format(smarts))
    if sanitize:
        Chem.SanitizeMol(mol)
    return mol

def binary_to_mol(data, sanitize=True):
    try:
        mol = Chem.Mol(data)
    except RuntimeError:
        raise ValueError("Invalid binary mol data: `{0}`".format(data))
    if sanitize:
        Chem.SanitizeMol(mol)
    return mol

def ctab_to_mol(data, sanitize=True):
    mol = Chem.MolFromMolBlock(data, sanitize=sanitize, removeHs=sanitize)
    if mol is None:
        raise ValueError("Failed to parse CTAB")
    return mol

def inchi_to_mol(inchi, sanitize=True):
    mol = Chem.MolFromInchi(inchi, sanitize=sanitize, removeHs=sanitize)
    if mol is None:
        raise ValueError("Failed to parse InChI: `{0}`".format(inchi))
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

def attempt_mol_coersion(data, sanitize=True):
     # RDKit doesn't like Unicode
    if isinstance(data, (basestring, buffer)):
        data = str(data)
    
    # Record all parsing errors
    errors = []
    
    # Try all known mol parsers
    for fmt, parser in MOL_PARSERS:
        try:
            mol = parser(data, sanitize=sanitize)
            return fmt, mol
        except ValueError as error:
            errors.append(str(error))
    raise ValueError("Failed to convert `{0}` to mol. Errors were: {1}".format(data, ", ".join(errors)))

def coerse_to_mol(data, sanitize=True):
    fmt, mol = attempt_mol_coersion(data, sanitize=sanitize)
    return mol
    
def infer_mol_format(data, sanitize=True):
    fmt, mol = attempt_mol_coersion(data, sanitize=sanitize)
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
        raise ValueError("BFP size does not match expected {0}".format(size))
    return value


def extract_bfp_element(value, size=None):
    if isinstance(value, BfpElement):
        value = value.as_bfp
    else:
        raise ValueError("Not already a bfp element")
    if size is not None and size != value.GetNumBits():
        raise ValueError("BFP size does not match expected {0}".format(size))
    return value
    

def bfp_from_raw_binary_text(raw, size):
    vect = DataStructs.CreateFromBinaryText(raw)
    if vect.GetNumBits() != size:
        raise ValueError("BFP size does not match expected {0}".format(size))
    return vect


def bfp_to_raw_binary_text(bfp):
    return bfp.ToBinary()
    

def bytes_from_binary_text(binary_text):
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
        size = len(bits)
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
    vect = bfp_from_bits(bits, size=size)
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

BFP_PARSERS = [
    ('bfp', ensure_bfp),
    ('element', extract_bfp_element),
    ('bytes', bfp_from_bytes),
    ('chars', bfp_from_chars),
    ('binary_text', bfp_from_binary_text),
    ('bits', bfp_from_bits),
    ('binary', bfp_from_raw_binary_text),
    # TODO: base64, etc.
]

def attempt_bfp_conversion(data, size=None, method=None):
    if isinstance(data, (basestring, buffer)):
        data = str(data)

    # Special case for mol elements
    if isinstance(data, Chem.Mol):
        if method is not None:
            data = method(data)
        else:
            raise ValueError("Attempting to generate bfp from Mol "
                             "but no method provided")
        
    errors = []
    
    # Try all known bfp parsers
    for fmt, parser in BFP_PARSERS:
        try:
            mol = parser(data, size=size)
            return fmt, mol
        except ValueError as error:
            errors.append(str(error))
        except TypeError as error:
            errors.append(str(error))
    raise ValueError("Failed to convert `{0}` to bfp. Errors were: {1}".format(data, ", ".join(errors)))
    
def coerse_to_bfp(data, size=None, method=None):
    fmt, bfp = attempt_bfp_conversion(data, size=size, method=None)
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
        'local_function': staticmethod(local_function),
        '_as_property': as_property,
        '_sql_cast_out': sql_cast_out,
        '_local_in_type': local_in_type,
        '_local_kwargs': local_kwargs,
        '_element_kwargs': element_kwargs,
    }

    docs = [help]
    docs.append(getattr(tpl, 'HELP', ''))

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
                    help="", register=True):
    return _hybrid_function(sql_function, local_function, 
                            as_property=as_property, 
                            type_=type_, 
                            local_in_type=local_in_type, 
                            sql_cast_out=sql_cast_out,
                            tpl=_RDKitFunction,
                            help=help, register=register)


@compiles(_RDKitFunction)
def _compile_rdkit_function(element, compiler, **kwargs):
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
        return hasattr(self._get_function(fn_name), 'local_function')

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
        return self.desc
    def __repr__(self):
        return "<%s at 0x%x; %r>" % (self.__class__.__name__,
                                        id(self), self.desc)


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
                                  expression.ColumnElement)):
            return super(_RDKitDataElement, self)\
                        ._function_call(fn_name, data=self.desc)

        elif self._can_call_function_locally(fn_name):
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
        return self._function_call(name)
        
    @property
    def desc(self):
        if isinstance(self.data, expression.BindParameter):
            return self.data
        else:
            return self.compile_desc_literal()
        
    def compile_desc_literal(self):
        raise NotImplemented

    def to_local_type(self):
        raise NotImplemented

    def new_with_attrs(self, data):
        cls = type(self)
        params = self._get_params()
        new = cls(data, **params)
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
    inchi = instrumented_property('inchi')
    inchi_key = instrumented_property('inchi_key')

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
    
    @staticmethod
    def mol_cast(self, value):
        return "{data}::mol".format(data=value)
        
    @property
    def as_mol(self):
        # Try and convert to rdkit.Chem.Mol (if not already a Chem.Mol)
        # Cache the mol to persist props and prevent recomputation
        if self._mol is None:
            self._mol = coerse_to_mol(self.data, sanitize=self.force_sanitized)
        return self._mol
    
    @property
    def as_binary(self):
        return self.as_mol.ToBinary()
    
    @property
    def as_smiles(self):
        return Chem.MolToSmiles(self.as_mol, isomericSmiles=True)
    
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
    

class _ExplicitMolElement(RawMolElement, expression.Function):
    """ Define a mol that expects a specific IO format """
    def __init__(self, mol, _force_sanitized=True):        
        RawMolElement.__init__(self, mol, _force_sanitized)
        expression.Function.__init__(self, self.backend_in, self.desc,
                                           type_=Mol(coerse_=self.frontend_coerse))
        
    @property
    def backend_in(self): 
        raise NotImplementedError
        
    @property
    def backend_out(self): 
        raise NotImplemented
    
    @property
    def frontend_coerse(self): 
        raise NotImplemented
        

class BinaryMolElement(_ExplicitMolElement):   
    backend_in = 'mol_from_pkl'
    backend_out = 'mol_to_pkl'
    frontend_coerse = 'binary'

    def compile_desc_literal(self):
        return self.as_binary
    

class SmartsMolElement(_ExplicitMolElement):   
    backend_in = 'mol_from_smarts'
    backend_out = 'mol_to_smarts'
    frontend_coerse = 'smarts'
    
    def compile_desc_literal(self):
        return self.as_smarts
    
    
class SmilesMolElement(_ExplicitMolElement):   
    backend_in = 'mol_from_smiles'
    backend_out = 'mol_to_smiles'
    frontend_coerse = 'smiles'
    
    def compile_desc_literal(self):
        return self.as_smiles
    
class CtabMolElement(_ExplicitMolElement):
    backend_in = 'mol_from_ctab'
    backend_out = 'mol_to_ctab'
    frontend_coerse = 'ctab'
    
    def compile_desc_literal(self):
        return self.as_ctab


## Binary fingerprint elements
## Handle conversion from various formats (some RDKit based and others)


class BfpElement(_RDKitDataElement, expression.Function, RDKitBfpProperties):
    
    backend_in = 'bfp_from_binary_text'
    backend_out = 'bfp_to_binary_text'

    def __init__(self, fp, size_=None, method_=None):
        _RDKitDataElement.__init__(self, fp)
        self._size = size_
        self._method = method_
        self._fp = None
        expression.Function.__init__(self, self.backend_in, self.desc,
                                           type_=Bfp(size=self._size,
                                                     method=self._method))

    def _get_params(self):
        return {
            'size_': self._size,
            'method_': self._method,
        }

    def compile_desc_literal(self):
        return self.as_binary_text

    def to_local_type(self):
        return self.as_bfp
    
    @property
    def as_bfp(self):
        if self._fp is None:
            self._fp = coerse_to_bfp(self.data, 
                                     size=self._size,
                                     method=self._method)
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
        return self._element is not None\
               and not isinstance(obj, self._element)

    def _cast_other_element(self, obj):
        raise NotImplemented

    @staticmethod
    def _ensure_other_element(fn):
        @functools.wraps(fn)
        def wrapper(self, other):
            other = self._cast_other_element(other)
            return fn(self, other)
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
            return self.force_sanitize

    def _cast_other_element(self, obj):
        if self._should_cast(obj):
            fmt = infer_mol_format(obj, sanitize=self._sanitize)
            convert = self.COERSIONS.get(fmt, RawMolElement)
            other = convert(obj, _force_sanitized=self._sanitize)
        else:
            other = obj
        return other

    @_RDKitComparator._ensure_other_element
    def structure_is(self, other):
        return self.op('@=')(other)

    @_RDKitComparator._ensure_other_element
    def has_substructure(self, other):
        return self.op('@>')(other)

    @_RDKitComparator._ensure_other_element
    def in_superstructure(self, other):
        return self.op('<@')(other)

    def contains(self, other, escape=None):
        return self.has_substructure(other)

    def contained_in(self, other):
        return self.in_superstructure(other)
        
    def __eq__(self, other):
        return self.structure_is(other)

    def __contains__(self, other):
        return self.has_substructure(other)
        
    def __getitem___(self, other):
        return self.has_substructure(other)


## SQLAlchemy data types


class Mol(UserDefinedType):
    name = 'mol'
    
    def __init__(self, coerse_='smiles', sanitized_=True):
        self.coerse = coerse_
        self.sanitized = sanitized_

    comparator_factory = _RDKitMolComparator

    def _coerse_compared_value(self, op, value):
        return self
    
    def _get_coersed_element(self, default=None):
        return self.comparator_factory.COERSIONS.get(self.coerse, default)
    
    def get_col_spec(self):
        return self.name
    
    def bind_expression(self, bindvalue):
        sanitize = self.sanitized
        element_type = self._get_coersed_element(default=RawMolElement)
        element = element_type(bindvalue, _force_sanitized=sanitize)
        return element
    
    def column_expression(self, col):
        element = self._get_coersed_element()
        if element is not None:
            fn_name = element.backend_out
            fn = getattr(expression.func, fn_name)
            return fn(col, type_=self)
        else:
            return RawMolElement.mol_cast(col)
    
    def bind_processor(self, dialect):
        def process(value):
            if isinstance(value, RawMolElement):
                return value.desc
            else:
                return value  # This may need to do further coersion
        return process
    
    def result_processor(self, dialect, coltype):
        element = self._get_coersed_element(default=RawMolElement)
        def process(value):
            if value is not None:
                return element(value, _force_sanitized=self.sanitized)
            else:
                return None
        return process


class BinaryMol(Mol):
    def __init__(self, coerse_='binary', sanitized_=True):
        super(BinaryMol, self).__init__(coerse_=coerse_, 
                                        sanitized_=sanitized_)


class _RDKitBfpComparator(_RDKitComparator,
                          _RDKitInstrumentedFunctions,
                          RDKitBfpProperties):
    _element = BfpElement
    default_coefficient = 'tanimoto'

    @property
    def active_coefficient(self):
        try:
            return self.coefficient
        except AttributeError:
            return self.default_coefficient

    def _cast_other_element(self, obj):
        if self._should_cast(obj):
            size = getattr(self, 'size', None)
            method = getattr(self, 'method', None)
            other = coerse_to_bfp(obj, size=size, method=method)
        else:
            other = obj
        return other

    @_RDKitComparator._ensure_other_element
    def tanimoto_similar(self, other):
        return self % other_element
    
    @_RDKitComparator._ensure_other_element
    def dice_similar(self, other):
        return self.op('#')(other_element)
    
    @_RDKitComparator._ensure_other_element
    def tanimoto_nearest_neighbors(self, other):
        return self.op('<%>')(other_element)

    @_RDKitComparator._ensure_other_element
    def dice_nearest_neighbors(self, other):
        return self.op('<#>')(other_element)

    def similar_to(self, other):
        coeff = self.active_coefficient
        other_element = self._get_other_element(other)
        similarity_fn = "{0}_similar".format(coeff)
        return getattr(self, similarity_fn)(other_element)

    def most_similar(self, other):
        coeff = self.active_coefficient
        other_element = self._get_other_element(coeff)
        similarity_fn = "{0}_nearest_neighbors".format(default)
        return getattr(self, similarity_fn)(other_element)


class Bfp(UserDefinedType):
    name = 'bfp'
    element = BfpElement
    size = None
    method = None

    comparator_factory = _RDKitBfpComparator
    
    def __init__(self, size=None, method=None, coefficient='tanimoto'):
        self.size = size or self.size
        self.method = method or self.method
        self.coefficient = coefficient
    
    def get_col_spec(self):
        return self.name
    
    def bind_expression(self, bindvalue):
        element = self.element(bindvalue, 
                               size_=self.size, 
                               method_=self.method)
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
                return self.element(value, size_=self.size)
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
                help="Returns the AMW for a molecule.")
    logp = _rdkit_function(
                'mol_logp', 
                Descriptors.MolLogP,
                help="Returns the LogP for a molecule.")
    tpsa = _rdkit_function(
                'mol_tpsa', 
                Descriptors.TPSA,
                help="Returns the topological polar surface  area for a "
                     "molecule.")
    hba = _rdkit_function(
                'mol_hba', 
                Descriptors.NumHAcceptors,
                help="Returns the number of Lipinski H-bond acceptors for a "
                     "molecule")
    hbd = _rdkit_function(
                'mol_hbd', 
                Descriptors.NumHDonors,
                help="Returns the number of Lipinski H-bond donors for a "
                     "molecule")
    num_atoms = _rdkit_function(
                'mol_numatoms', 
                Chem.Mol.GetNumAtoms,
                help="Returns the number of atoms in a molecule")
    num_heavy_atoms = _rdkit_function(
                'mol_numheavyatoms', 
                Chem.Mol.GetNumHeavyAtoms,
                help="Returns the number of heavy atoms in a molecule")
    num_rotatable_bonds = _rdkit_function(
                'mol_numrotatablebonds', 
                Descriptors.NumRotatableBonds,
                help="Returns the number of rotatable bonds in a molecule")
    num_hetero_atoms = _rdkit_function(
                'mol_numheteroatoms', 
                Descriptors.NumHeteroatoms,
                help="Returns the number of heteroatoms in a molecule")
    num_rings = _rdkit_function(
                'mol_numrings', 
                Descriptors.RingCount,
                help="Returns the number of rings in a molecule")
#    num_aromatic_rings = _rdkit_function('mol_numaromaticrings', Mol,
#                               "Returns the number of aromatic rings "
#                               "in a molecule")
#    num_aliphatic_rings = _rdkit_function('mol_numaliphaticrings', Mol,
#                               "Returns the number of aliphatic rings "
#                               "in a molecule")
#    num_saturated_rings = _rdkit_function('mol_numsaturatedrings', Mol,
#                               "Returns the number of saturated rings "
#                               "in a molecule")
#    num_saturated_rings = _rdkit_function('mol_numsaturatedrings', Mol,
#                               "Returns the number of saturated rings "
#                               "in a molecule")
#    num_aromaticheterocycles = _rdkit_function('mol_numaromaticheterocycles', Mol,
#                               "Returns the number of aromatic heterocycles "
#                               "in a molecule")
#    formula = _rdkit_function('mol_formula', Mol,
#            'Returns a string with the molecular formula. The second '
#            'argument controls whether isotope information is '
#            'included in the formula; the third argument controls '
#            'whether "D" and "T" are used instead of [2H] and [3H].')
#
#    chi0v = _rdkit_function('mol_chi0v', Mol,
#            'Returns the ChiVx value for a molecule for X=0-4')
#    chi0n = _rdkit_function('mol_chi0n', Mol,
#            'Returns the ChiVx value for a molecule for X=0-4')
#    kappa1 = _rdkit_function('mol_kappa1', Mol,
#            'Returns the kappaX value for a molecule for X=1-3')

    inchi = _rdkit_function(
                'mol_inchi', 
                Chem.MolToInchi,
                sql_cast_out='text',
                help='Returns an InChI for the molecule. (available '
                     'from the 2011_06 release, requires that the RDKit be '
                     'uilt with InChI support).')

    inchi_key = _rdkit_function(
                'mol_inchikey', 
                lambda m: Chem.InchiToInchiKey(Chem.MolToInchi(m)),
                sql_cast_out='text',
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


def _rdkit_bfp_binary_fn(fn):
    def wrapped(a, b, *args, **kwargs):
        return fn(coerse_to_bfp(a), coerse_to_bfp(b), *args, **kwargs)
    return wrapped


class _RDKitBfpFunctions(object):
    size = _rdkit_function(
                'size',
                DataStructs.ExplicitBitVect.GetNumBits,
                help="Returns the number of bits in a binary fingerprint")

    tanimoto = _rdkit_function(
                'tanimoto_sml',
                _rdkit_bfp_binary_fn(DataStructs.TanimotoSimilarity),
                as_property=False)

    dice = _rdkit_function(
                'dice_sml',
                _rdkit_bfp_binary_fn(DataStructs.DiceSimilarity),
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

    def set(self, value):
        value = self.type_(value)
        query = 'SET {variable}=:value'.format(variable=self.variable)
        xpr = expression.text(query)
        return xpr.execution_options(autocommit=True)\
                  .params(value=value)

    def get(self):
        query = 'SHOW {variable}'.format(variable=variable)
        return expression.text(query)

    @contextlib.contextmanager
    def set_in_context(self, value):
        original = self.get()
        self.set(value)
        yield
        self.set(original)

    def __call__(self, value):
        return self.set_in_context(value)


@compiles(GUC)
def __compile_guc(element, compiler, **kwargs):
    return compiler.process(element.get())


tanimoto_threshold = GUC('rdkit.tanimoto_threshold', float)
dice_threshold = GUC('rdkit.dice_threshold', float)


class RDKitMolClass(RDKitInstrumentedColumnClass, RDKitMolProperties):
    pass


class RDKitBfpClass(RDKitInstrumentedColumnClass, RDKitBfpProperties):
    pass


## Define the rdkit datatypes in sqlalchemy

ischema_names['mol'] = Mol
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
