package require psfgen
topology top_all36_cgenff.rtf
topology top_lignin.top
segment L {
    residue 1 C
    residue 2 C
    residue 3 C
    residue 4 C
    residue 5 C
    residue 6 C
    residue 7 C
    residue 8 C
    residue 9 C
    residue 10 C
    residue 11 C
    residue 12 C
    residue 13 C
    residue 14 C
}
patch B5C L:7 L:1
patch B5C L:6 L:3
patch BO4 L:4 L:6
patch BO4 L:5 L:4
patch BO4 L:2 L:5
patch BO4 L:8 L:2
patch BO4 L:9 L:8
patch BO4 L:10 L:9
patch BO4 L:11 L:10
patch B5C L:12 L:13
patch BO4 L:14 L:12
regenerate angles dihedrals
writepsf lignin.psf
