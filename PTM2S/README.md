# PTM2S
This function provides the mapping of the predicted PTM sites to protein 3D structure.
- OS: Ubuntu 16.04.5 LTS
- Python: Python 3.5 

#### Run
```r
python3 ptm2Structure.py -ptmInput [FASTA input sequences] -ptmOutput [result file from MusiteDeep predictor] -o [output folder]
```

For example:
```r
python3 ptm2Structure.py -ptmInput test_seq.fasta -ptmOutput Prediction_results.txt -o ./ -maxPDB 2
```

For details of the parameters, use the -h or --help parameter.



#### Outputs:  
json file ptm2Structure.json in user specified output folder

ptm2Structure.json for the example:
```r
[{"ProteinId": ">sp|P97756|KKCC1_RAT Calcium/calmodulin-dependent protein kinase kinase 1 OS=Rattus norvegicus GN=Camkk1 PE=1 SV=1", 
  "PTM2Structure": 
  [{
    "pdbNo": "5uyj_A_1", 
    "pdbId": "5uyj", 
    "chain": "A", 
    "evalue": "1.06093E-146", 
    "bitscore": 424.091, 
    "identity": 196.0, 
    "identityPositive": 247.0, 
    "pdbFrom": 2, 
    "pdbTo": 291, 
    "seqFrom": 123, 
    "seqTo": 412, 
    "PTMannotation": ["308:S:S:Phosphoserine:0.847", "309:S:N:Phosphoserine:0.894", "313:T:T:Phosphothreonine:0.765", "328:S:I:Phosphoserine:0.51"]
    }, 
    {
     "pdbNo": "5uy6_A_1", 
     "pdbId": "5uy6", 
     "chain": "A", 
     "evalue": "2.28703E-141", 
     "bitscore": 410.223, 
     "identity": 197.0, 
     "identityPositive": 248.0, 
     "pdbFrom": 2, 
     "pdbTo": 290, 
     "seqFrom": 123, 
     "seqTo": 411, 
     "PTMannotation": ["308:S:S:Phosphoserine:0.847", "309:S:N:Phosphoserine:0.894", "313:T:T:Phosphothreonine:0.765", "328:S:I:Phosphoserine:0.51"]
     }
   ]}
```

The "PTMannotation" filed contains the infomation (colon-determined)for each position, the first is the queryPosition, the second is the queryAminoAcid, the third is the pdbAminoAcid, the last is the PTM prediction results for that position.