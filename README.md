### Prelims
- clone repository
```
git clone https://github.com/anhkhoado93/SpellCorrection.git

cd SpellCorrection
mkdir weights
mkdir weights/model
mkdir weights/history
```
- download [this folder](https://drive.google.com/drive/folders/1v2AHP3tsDFimp72AjrwVIM-WLf9KS4gW?usp=sharing) and put it in **SpellCorrection/input** folder. Or manually download each file:

```
cd SpellCorrection/input
mkdir luanvan
cd luanvan
gdown "https://drive.google.com/uc?id=1-BNP5QIXOXckeOCzD3koAb2cLSL8Mi1e"
gdown "https://drive.google.com/uc?id=1-7BTdejL03-Rx4mRfkOeTgK1Nl4T9bAv&confirm=t"
gdown "https://drive.google.com/uc?id=1-L8F9kGnV_Ob0B3aHd-fEdfavd_y3s8h"
gdown "https://drive.google.com/uc?id=1-E9wdKWpoNeYrze9JZLbqSiWp9FORiQF&confirm=t"
gdown "https://drive.google.com/uc?id=1-Kq0joUm4_XVqMnJYJ1HvcfKr5-sJRo_"
gdown "https://drive.google.com/uc?id=1-67HkPgxKrtDXB0325DDgKM0vX6011G4"
```

- download requirements
```
cd SpellCorrection
pip install -r requirements.txt
```

### Train
```
python train.py
```