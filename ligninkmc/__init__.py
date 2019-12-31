try:
	from ligninkmc.Visualization import generateMol, moltosvg
except Exception as e:
	print(e)
	print("\nRDKit is likely not installed correctly. Visualization methods will be unavailable.")