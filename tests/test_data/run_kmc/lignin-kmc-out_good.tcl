package require psfgen
topology toppar/top_all36_cgenff.rtf
topology toppar/top_lignin.top
segment L {
    residue 1 GUAI
    residue 2 GUAI
    residue 3 GUAI
    residue 4 SYR
    residue 5 SYR
    residue 6 SYR
    residue 7 GUAI
    residue 8 SYR
    residue 9 GUAI
    residue 10 SYR
}
patch B5G L:1 L:2
patch BO4 L:3 L:1
patch B5G L:4 L:3
patch BO4 L:5 L:4
patch BO4 L:6 L:5
patch BO4 L:7 L:6
patch BO4 L:8 L:7
patch BO4 L:9 L:8
patch B5G L:10 L:9
regenerate angles dihedrals
writepsf lignin.psf
