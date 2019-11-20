package require psfgen
topology top_all36_cgenff.rtf
topology top_lignin.top
segment 1 {
    residue 1 G
    residue 2 G
    residue 3 G
    residue 4 G
}
patch BB 1:1 1:4
patch BO4 1:2 1:4
patch B5G 1:3 1:1
regenerate angles dihedrals
writepsf birch.psf
