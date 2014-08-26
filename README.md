rdalchemy
=========

RDKit integration to SQLAlchemy

Example Usage
=============

```python
    from sqlalchemy import (ceate_engine, Column, Integer, String, ForeignKey)
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.ext.declarative import declarative_base
    
    from rdkit.Chem import RDKFingerprint
    
    from rdalchemy import (BinaryMol, Bfp, tanimoto_threshold)
    
    engine = create_engine('postgresql+psycopg2://readonly:access@db/zinc', echo=True)
    metadata = MetaData(engine)
    Base = declarative_base(metadata=metadata)
    
    
    class Substance(Base):
      __tablename__ = 'substance'
        
        sub_id = Column('sub_id', Integer, primary_key=True)
        structure = Column('smiles', BinaryMol)
        name = Column('name', String)
        fingerprint = Column('fp_id', ForeignKey('Fingerprint.fp_id'))
        
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
            
    class Fingerprint(Base):
      __tablename__ = 'fingerprints'
      
      SIZE = 2048
      MAX_PATH = 5
      
      @staticmethod
      def method(mol):
        if hasattr(mol, 'as_mol'):
            mol = mol.as_mol
        return RDKFingerprint(mol, maxPath=MAX_PATH, fpSize=SIZE)
      
      fp_id = Column('fp_id', Integer, primary_key=True)
      data = Column('data', Bfp(size=SIZE, method=method))
      
      @property
      def as_array(self):
        return self.data.as_array
    
      @classmethod
      def similar_to(cls, value):
        return cls.data.similar_to(value)

    db = sessionmaker(bind=engine)()
    
    smiles = 'CC(=O)Oc1cccccc1C(=O)[O-]'
    aspirin = db.query(Substance).filter(Substance.structure==smiles).one()
    
    print aspirin.smiles
    print aspirin.structure.as_pdb
    
    similar = db.query(Substance).join(Fingerprint)\
                .filter(Fingerprint.similar_to(aspirin))
    with tanimoto_threshold(.7):
        for similar_substance in similar:
            print similar_substance.smiles, similar_substance.sub_id
    
    
    tenbenz = db.query(Substance)\
                .filter(Substance.structure.contains('c1ccccc1'))\
                .limit(10)
    
    for substance in tenbenz:
        print substance.sub_id, substance.smarts
```
