package require psfgen
topology top_all36_cgenff.rtf
topology top_lignin.top
segment 1 {
    residue 1 G
    residue 2 G
    residue 3 G
    residue 4 S
}
patch BB 1:2 1:4
patch BB 1:1 1:3
patch B5C 1:2 1:1
regenerate angles dihedrals
writepsf birch.psf
