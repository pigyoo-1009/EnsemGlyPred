def read_fasta_file(fasta_file):
    with open(fasta_file, 'r') as f:
        lines = f.readlines()

    sequences = []
    sequence = ''
    for line in lines:
        if line.startswith('>'):
            if sequence:
                sequences.append(sequence)
                sequence = ''
        else:
            sequence += line.strip()
    if sequence:
        sequences.append(sequence)
    return sequences