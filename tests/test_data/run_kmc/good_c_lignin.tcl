package require psfgen
topology top_all36_cgenff.rtf
topology top_lignin.top
segment L {
    residue 1 CAT
    residue 2 CAT
    residue 3 CAT
    residue 4 CAT
    residue 5 CAT
    residue 6 CAT
    residue 7 CAT
    residue 8 CAT
    residue 9 CAT
    residue 10 CAT
    residue 11 CAT
    residue 12 CAT
    residue 13 CAT
    residue 14 CAT
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
