package require psfgen
topology top_all36_cgenff.rtf
topology top_lignin.top
segment 1 {
    residue 1 GUAI
    residue 2 GUAI
    residue 3 GUAI
    residue 4 SYR
}
patch B5G 1:4 1:3
patch BB 1:1 1:2
patch 4O5 1:4 1:2
regenerate angles dihedrals
writepsf birch.psf
