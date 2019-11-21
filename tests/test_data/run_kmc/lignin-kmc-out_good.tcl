package require psfgen
topology toppar/top_all36_cgenff.rtf
topology toppar/top_lignin.top
segment L {
    residue 1 G
    residue 2 S
    residue 3 S
}
patch B5G L:2 L:1
regenerate angles dihedrals
writepsf lignin.psf
