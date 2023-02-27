import sys

infilename = sys.argv[1]
mode = int(sys.argv[2])

if mode == 1:
    print("Mode chosen: source-side unique only")
    fw = open(infilename + '.srcuniq', 'w')
elif mode == 2:
    print("Mode chosen: 1-to-1")
    fw = open(infilename + '.1to1', 'w')
else: 
    print("Please choose a mode. Exiting.")
    exit()

fr = open(infilename, 'r')

all_en=set()
all_de=set()
output_lines = []
for line in fr:
    en, de = line.split()
    en = en.strip()
    de = de.strip()
    if ((mode == 1 and en not in all_en) or 
            (mode == 2 and en not in all_en and de not in all_de)):
        all_en.add(en)
        all_de.add(de)
        output_lines.append("{0} {1}\n".format(en, de))
fw.writelines(output_lines)
fr.close()
fw.close()

