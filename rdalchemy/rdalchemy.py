#!/usr/bin/env python

import numpy as np
import types

from sqlalchemy import event, Table
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.sql import expression, functions, type_coerce
from sqlalchemy.types import UserDefinedType, _Binary, TypeDecorator
from sqlalchemy.dialects.postgresql.base import ischema_names

from rdkit import Chem, DataStructs


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

def attempt_bfp_conversion(data, size=None):
    if isinstance(data, (basestring, buffer)):
        data = str(data)
        
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
    
def coerse_to_bfp(data, size=None):
    fmt, bfp = attempt_bfp_conversion(data, size=size)
    return bfp


## Core Types #######################################################
## TODO: Some of this turned out to be unused. Clean up
## Functions need a lot of work. Still need to figure that one out
## Consult geoalchemy2 and razi for possible solutions


class _RDKitType(UserDefinedType):
    def __getattr__(self, name):
        print name


class _RDKitFunction(functions.GenericFunction):
    
    def __init__(self, *args, **kwargs):
        expr = kwargs.pop('expr', None)
        if expr is not None:
            args = (expr,) + args
        super(RDKitFunction, self).__init__(*args, **kwargs)

    FN_NAME_PREFIX = "rdkit_"
    HELP_URL = 'See http://www.rdkit.org/docs/Cartridge.html'

    @classmethod
    def define(cls, 
               psql_name, 
               local_fn=None,
               type_=None, 
               help="", 
               register=True):

        attributes = {
            'name': name,
            'local_function': local_fn,
        }
        docs = [help, cls.HELP_URL]

        if type_ is not None:
            attributes['type'] = type_
            type_str = type_.__module__ + '.' + type_.__name__

        attributes['__doc__'] = "\n\n".join(docs)

        cls_name = name
        sub_cls = type(cls_name, (cls,), attributes)

        if register:
            globals()[cls_name] = sub_cls

        return sub_cls


class _RDKitFunctionCollection(object):
    @classmethod
    def _inject(cls, target):
        fns = [name for name in dir(cls) if not name.startswith('_')]
        for fn_local_name in fns:
            fn = getattr(cls, fn_local_name)
            attr = cls._make_function_attribute(fn, target)
            setattr(target, fn_local_name, attr)
            
    @classmethod
    def _make_function_attribute(cls, fn, target):
        def wrapper(self):
            func_gen = functions._FunctionGenerator(expr=self)
            func_ = getattr(func_gen, fn.name)
            return func_
        method = types.MethodType(wrapper, None, target)
        attribute = property(method)
        return attribute
        

class _RDKitElement(object):

    def __str__(self):
        return self.desc
    def __repr__(self):
        return "<%s at 0x%x; %r>" % (self.__class__.__name__,
                                        id(self), self.desc)
    

class _RDKitDataElement(_RDKitElement):
    """ Base datatype for object that need explicit modification
        either into or out of psql.
        Define "compile_desc_literal" to convert
    """
    
    @classmethod
    def _assign_function_object(cls, fns):
        cls._functions = {}
        fn_names = [fn for fn in dir(fns) if not fn.startswith('_')]
        for fn_name in fn_names:
            fn = getattr(fns, fn_name)
            cls._functions[fn_name] = fn
    
    def __init__(self, data):
        self.data = data

    def _function_call(self, fn_name):
        fn = self._functions[fn_name]
        if isinstance(self.data, expression.BindParameter):
            return fn(self.data)
        else:
            local = fn.local_function
            
        
    @property
    def desc(self):
        if isinstance(self.data, expression.BindParameter):
            return self.data
        else:
            return self.compile_desc_literal()
        
    def compile_desc_literal(self):
        raise NotImplemented


## Mol element interface.
## Either implicit (punt to postgres) or explicit (define functions
## to perfrom in/out conversions)


class RawMolElement(_RDKitDataElement):
    """ Base mol element. Let postgres deal with it. 
        Also define explicit conversion methods for mols"""
    def __init__(self, mol, _force_sanitized=True):
        _RDKitDataElement.__init__(self, mol)
        self._mol = None
        self.force_sanitized = _force_sanitized
    
    def compile_desc_literal(self):
        return self.mol_cast(self.data)
    
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
        raise NotImplemented
        
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


class BfpElement(_RDKitDataElement, expression.Function):
    
    backend_in = 'bfp_from_binary_text'
    backend_out = 'bfp_to_binary_text'

    def __init__(self, fp, size_=None):
        _RDKitDataElement.__init__(self, fp)
        self._size = size_
        self._fp = None
        expression.Function.__init__(self, self.backend_in, self.desc,
                                           type_=Bfp(size=self._size))
    
    def compile_desc_literal(self):
        return self.as_binary_text
    
    @property
    def size(self):
        return self.as_bfp.GetNumBits()
    
    @property
    def as_bfp(self):
        if self._fp is None:
            self._fp = coerse_to_bfp(self.data, size=self._size)
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


## SQLAlchemy data types


class Mol(UserDefinedType):
    name = 'mol'
    
    COERSIONS = {
        'smiles': SmilesMolElement,
        'smarts': SmartsMolElement,
        'binary': BinaryMolElement,
        'ctab': CtabMolElement,

        # Special Case Explicit Mol Elements
        'mol': BinaryMolElement,
    }
    
    def __init__(self, coerse_='smiles', sanitized_=True):
        self.coerse = coerse_
        self.sanitized = sanitized_

    class comparator_factory(UserDefinedType.Comparator):
        force_sanitize = False
        
        def _get_other_element(self, other):
            if not isinstance(other, RawMolElement):
                sanitize = self.force_sanitize
                fmt = infer_mol_format(other, sanitize=sanitize)
                element_type = Mol.COERSIONS.get(fmt, RawMolElement)
                other = element_type(other, _force_sanitized=sanitize)
            return other
        
        def structure_is(self, other):
            other_element = self._get_other_element(other)
            return self.op('@=')(other_elemenet)
        
        def has_substructure(self, other):
            other_element = self._get_other_element(other)
            return self.op('@>')(other_element)
        
        def in_superstructure(self, other):
            other_element = self._get_other_element(other)
            return self.op('<@')(other_element)
        
        def contains(self, other, **kwargs):
            return self.has_substructure(other)
        
        def __contains__(self, other):
            return self.has_substructure(other)
        
        def __getitem___(self, other):
            return self.has_substructure(other)
        
    def _coerse_compared_value(self, op, value):
        return self
    
    def _get_coersed_element(self, default=None):
        return self.COERSIONS.get(self.coerse, default)
    
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


class Bfp(UserDefinedType):
    name = 'bfp'
    element = BfpElement
    
    def __init__(self, size=None):
        self.size = size
    
    class comparator_factory(UserDefinedType.Comparator):
        
        default_similarity = 'tanimoto'
        
        def _get_other_element(self, other):
            if not isinstance(other, Bfp.element):
                other = Bfp.element(coerse_to_bfp(other))
            return other
        
        def tanimoto_similar(self, other):
            other_element = self._get_other_element(other)
            return self % other_element
        
        def dice_similar(self, other):
            other_element = self._get_other_element(other)
            return self.op('#')(other_element)
        
        def similar_to(self, other):
            default = self.default_similarity
            other_element = self._get_other_element(other)
            similarity_fn = "{0}_similar".format(default)
            return getattr(self, similarity_fn)(other_element)
    
    def get_col_spec(self):
        return self.name
    
    def bind_expression(self, bindvalue):
        element = self.element(bindvalue, size_=self.size)
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

## TODO: Fix these to be accessible as attributes of Mol/BFP objects
##       as well as elements. Will need to also define the 'local' 
##       equivalents of each function to make mol weight computable
##       in the same way without touching the database.


#class _RDKit_Mol_Functions(_RDKitFunctionCollection):
#    # Descriptors
#    amw = _RDKitFunction.define('mol_amw', Mol,
#                               "Returns the AMW for a molecule.")
#    logp = _RDKitFunction.define('mol_logp', Mol,
#                                "Returns the LogP for a molecule.")
#    tpsa = _RDKitFunction.define('mol_tpsa', Mol,
#                                "Returns the topological polar "
#                                "surface  area for a molecule.")
#    hba = _RDKitFunction.define('mol_hba', Mol,
#                               "Returns the number of Lipinski "
#                               "H-bond acceptors for a molecule")
#    hbd = _RDKitFunction.define('mol_hbd', Mol,
#                               "Returns the number of Lipinski "
#                               "H-bond donors for a molecule")
#    num_atoms = _RDKitFunction.define('mol_numatoms', Mol,
#                               "Returns the number of atoms in "
#                               "a molecule")
#    num_heavy_atoms = _RDKitFunction.define('mol_numheavyatoms', Mol,
#                               "Returns the number of heavy atoms "
#                               "in a molecule")
#    num_rotatable_bonds = _RDKitFunction.define('mol_numrotatablebonds', Mol,
#                               "Returns the number of rotatable "
#                               "bonds in a molecule")
#    num_hetero_atoms = _RDKitFunction.define('mol_numheteroatoms', Mol,
#                               "Returns the number of heteroatoms "
#                               "in a molecule")
#    num_rings = _RDKitFunction.define('mol_numrings', Mol,
#                               "Returns the number of rings "
#                               "in a molecule")
#    num_aromatic_rings = _RDKitFunction.define('mol_numaromaticrings', Mol,
#                               "Returns the number of aromatic rings "
#                               "in a molecule")
#    num_aliphatic_rings = _RDKitFunction.define('mol_numaliphaticrings', Mol,
#                               "Returns the number of aliphatic rings "
#                               "in a molecule")
#    num_saturated_rings = _RDKitFunction.define('mol_numsaturatedrings', Mol,
#                               "Returns the number of saturated rings "
#                               "in a molecule")
#    num_saturated_rings = _RDKitFunction.define('mol_numsaturatedrings', Mol,
#                               "Returns the number of saturated rings "
#                               "in a molecule")
#    num_aromaticheterocycles = _RDKitFunction.define('mol_numaromaticheterocycles', Mol,
#                               "Returns the number of aromatic heterocycles "
#                               "in a molecule")
#    formula = _RDKitFunction.define('mol_formula', Mol,
#            'Returns a string with the molecular formula. The second '
#            'argument controls whether isotope information is '
#            'included in the formula; the third argument controls '
#            'whether "D" and "T" are used instead of [2H] and [3H].')
#
#    chi0v = _RDKitFunction.define('mol_chi0v', Mol,
#            'Returns the ChiVx value for a molecule for X=0-4')
#    chi0n = _RDKitFunction.define('mol_chi0n', Mol,
#            'Returns the ChiVx value for a molecule for X=0-4')
#    kappa1 = _RDKitFunction.define('mol_kappa1', Mol,
#            'Returns the kappaX value for a molecule for X=1-3')
#
#    inchi = _RDKitFunction.define('mol_inchi', Mol,
#            'Returns an InChI for the molecule. (available '
#            'from the 2011_06 release, requires that the RDKit be '
#            'uilt with InChI support).')
#
#    inchi_key = _RDKitFunction.define('mol_inchikey', Mol,
#            'Returns an InChI key for the molecule. (available '
#            'from the 2011_06 release, requires that the RDKit be '
#            'uilt with InChI support).')


## Code to handle modifying the similarity search threshold constants 
## Borrowed straight from RAZI

def identity(x): 
    return x

class GUC(expression.Executable, expression.ClauseElement):
    """ From: Razi """

    def __init__(self, variable, type_=identity, *args, **kwargs):
        self.variable = variable
        self.type_ = type_

    def set(self, value):
        value = self.type_(value)
        query = 'SET {variable}=:value'.format(variable=self.variable)
        xpr = expression.text(query)
        return xpr.execution_options(autocommit=True)\
                  .params(value=value)

    def get(self):
        query = 'SHOW {variable}'.format(variable=variable)
        return expression.text(query)


@compiles(GUC)
def __compile_guc(element, compiler, **kwargs):
    return compiler.process(element.get())


tanimoto_threshold = GUC('rdkit.tanimoto_threshold', float)
dice_threshold = GUC('rdkit.dice_threshold', float)


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
