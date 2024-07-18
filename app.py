import json
import pandas as pd

from deepface import DeepFace

demographies=DeepFace.analyze(img_path = "elon.jpg", detector_backend='retinaface' )

df = pd.DataFrame(demographies)
print(df)
print(json.dumps(demographies, indent = 2))
