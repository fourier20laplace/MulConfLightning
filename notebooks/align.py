from openbabel import openbabel

# Create an Open Babel object for reading and writing
obConversion = openbabel.OBConversion()

ref_path = "/home/lmh/projects_dir/Antibody_Mutation/data/SKEMPIv2/PDBs_fixed/1A4Y.pdb"
conf_path = "/home/lmh/projects_dir/Antibody_Mutation/data/SKEMPIv2/PDBs_fixed/af3_sp2_output/1a4y_sp2_rand0.2/250419_233850/1a4y_sp2_rand0.2_250419_233850_model.cif"

# Create OBMol objects for reference and conformer molecules
obConversion.SetInFormat("pdb")
mol_ref = openbabel.OBMol()
obConversion.ReadFile(mol_ref, ref_path)

obConversion.SetInFormat("cif") 
mol = openbabel.OBMol()
obConversion.ReadFile(mol, conf_path)

# Create an OBAlign object for alignment
align = openbabel.OBAlign()

# Add the reference molecule and the molecule to be aligned
align.SetTarget(mol_ref)
align.AddMol(mol)

# Perform alignment
align.Align()

# Save the aligned molecule as PDB
obConversion.SetOutFormat("pdb")
aligned_pdb_path = "/home/lmh/projects_dir/af3/alphafold3/MyAnalyze/analyze_dir/tstopenbabel/aligned_conformer.pdb"
obConversion.WriteFile(mol, aligned_pdb_path)

print(f"Aligned molecule saved to {aligned_pdb_path}")
