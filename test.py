def encode_y(y):
    if '#+' in y:
        val = float(y.replace('#+',''))
        y = float(9999 - 100*(val-1))
    elif '#-' in y:
        val = float(y.replace('#-',''))
        y = float(-9999 + 100*(val-1))
    elif '+' in y or y == '0':
        y = float(y.replace('+',''))
    elif '-' in y:
        y = -float(y.replace('-',''))
    else:
        raise Exception('y Encoding Error')
    return float(y/9999)  # for normalising 

out = encode_y('-1')
print(out)
out = encode_y('+8433')
print(out)
out = encode_y('#+10')
print(out)
out = encode_y('#-1')
print(out)

import sklearn
print('Scikit-learn version:', sklearn.__version__)