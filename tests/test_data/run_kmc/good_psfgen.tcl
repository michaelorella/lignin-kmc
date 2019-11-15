package require psfgen
topology toppar/top_all36_cgenff.rtf
topology toppar/top_lignin.top
segment L {
    residue 1 S
    residue 2 S
    residue 3 G
    residue 4 G
    residue 5 G
    residue 6 G
    residue 7 G
    residue 8 G
    residue 9 G
    residue 10 G
}
patch BB L:1 L:2
patch BO4 L:3 L:1
patch BO4 L:4 L:2
patch BO4 L:6 L:5
patch BO4 L:8 L:7
patch B5G L:10 L:9
patch B5C L:10 L:3
patch B5C L:6 L:4
patch 4O4 L:10 L:8
regenerate angles dihedrals
writepsf L.psf
