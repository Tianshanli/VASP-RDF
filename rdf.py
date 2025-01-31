import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as mda
from MDAnalysis.analysis import rdf
from itertools import combinations_with_replacement
from matplotlib import colormaps

# ------------------------- XDATCAR to PDB Conversion -------------------------
def read_xdatcar(filename):
    """Read XDATCAR file and convert to Cartesian coordinates"""
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f.readlines()]

    scale = float(lines[1])
    lattice = np.array([[float(x)*scale for x in lines[i].split()] for i in range(2,5)])

    atom_types = lines[5].split()
    atom_counts = list(map(int, lines[6].split()))
    total_atoms = sum(atom_counts)

    positions = []
    current_line = 7
    while current_line < len(lines):
        if lines[current_line].startswith(('Direct','Cartesian')):
            coord_type = lines[current_line].split()[0]
            current_line += 1
            frame = []
            for _ in range(total_atoms):
                if current_line >= len(lines) or not lines[current_line]:
                    break
                coords = list(map(float, lines[current_line].split()[:3]))
                if coord_type == "Direct":
                    coords = np.dot(coords, lattice)  # Convert fractional coordinates to Cartesian
                frame.append(coords)
                current_line += 1
            positions.append(frame)
        else:
            current_line += 1

    return lattice, atom_types, atom_counts, positions

def write_pdb(filename, lattice, atom_types, atom_counts, positions):
    """Write to PDB file"""
    a, b, c = np.linalg.norm(lattice, axis=1)
    alpha = np.degrees(np.arccos(np.dot(lattice[1], lattice[2])/(b*c)))
    beta = np.degrees(np.arccos(np.dot(lattice[0], lattice[2])/(a*c)))
    gamma = np.degrees(np.arccos(np.dot(lattice[0], lattice[1])/(a*b)))

    with open(filename, 'w') as f:
        for frame_idx, frame in enumerate(positions, 1):
            f.write(f"MODEL     {frame_idx}\n")
            f.write(f"CRYST1{a:9.3f}{b:9.3f}{c:9.3f}{alpha:7.2f}{beta:7.2f}{gamma:7.2f} P1   1\n")
            
            atom_index = 1
            current_atom = 0
            for atom_type, count in zip(atom_types, atom_counts):
                for _ in range(count):
                    x, y, z = frame[current_atom]
                    f.write(f"ATOM  {atom_index:5d} {atom_type:3}  MOL     1    "
                            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {atom_type:2}\n")
                    atom_index += 1
                    current_atom += 1
            f.write("TER\nENDMDL\n")

# ------------------------- RDF Automatic Analysis -------------------------
def analyze_rdf_auto(pdb_file, r_max=6.0, bin_width=0.1):
    """Automatically identify all atom pairs and calculate RDF"""
    # Create Universe and get all element types
    u = mda.Universe(pdb_file)
    if u.dimensions is None:
        raise ValueError("PDB file is missing unit cell information")
    
    # Get all unique element types
    elements = list(set(u.atoms.elements))
    if len(elements) < 1:
        raise ValueError("No atomic data found")
    
    # Generate all possible atom pairs (ordered, including duplicates like Li-Li)
    pairs = list(combinations_with_replacement(elements, 2))
    if not pairs:
        raise ValueError("Unable to generate atom pairs")
    
    # Automatically assign colors (using matplotlib colormap)
    cmap = colormaps.get_cmap('tab10')
    colors = {pair: cmap(i % 10) for i, pair in enumerate(pairs)}
    
    # Compute RDF for each atom pair
    results = {}
    bins = int(r_max / bin_width)
    
    for pair in pairs:
        elem_A, elem_B = pair
        try:
            group_A = u.select_atoms(f"element {elem_A}")
            group_B = u.select_atoms(f"element {elem_B}")
            
            if len(group_A) == 0 or len(group_B) == 0:
                print(f"Skipping {elem_A}-{elem_B}: No atoms found")
                continue
                
            rdf_analyzer = rdf.InterRDF(group_A, group_B, 
                                      nbins=bins, 
                                      range=(0, r_max),
                                      exclusion_block=(1, 1))  # Exclude adjacent atoms
            rdf_analyzer.run()
            results[pair] = (rdf_analyzer.results.bins, rdf_analyzer.results.rdf)
        except Exception as e:
            print(f"Error calculating {elem_A}-{elem_B}: {str(e)}")
    
    # Plot
    plt.figure(figsize=(8, 6), dpi=300)
    for pair, (r, g_r) in results.items():
        label = f"{pair[0]}-{pair[1]}"
        plt.plot(r, g_r, label=label, color=colors[pair], linewidth=2)
    
    plt.title("Radial Distribution Functions", fontsize=14)
    plt.xlabel(r"Distance ($\AA$)", fontsize=12)
    plt.ylabel("g(r)", fontsize=12)
    plt.legend()
    plt.ylim(0, 5)
    plt.xlim(0, 6)
    plt.tight_layout()
    plt.savefig('rdf_vasp.png', bbox_inches='tight')
    plt.close()
    print("RDF plot saved as rdf.png")

# ------------------------- Main Program -------------------------
if __name__ == "__main__":
    # Convert XDATCAR to PDB
    input_file = "XDATCAR"
    output_file = "output.pdb"
    
    lattice, atom_types, atom_counts, positions = read_xdatcar(input_file)
    write_pdb(output_file, lattice, atom_types, atom_counts, positions)
    print(f"Conversion successful: {input_file} -> {output_file}")
    
    # Automatic RDF analysis
    analyze_rdf_auto(output_file, r_max=6.0, bin_width=0.1)
