from PIL import Image
import numpy as np
results = np.genfromtxt('../lib/errors.csv',delimiter=',')

for index in range(results.shape[0]):
    answer = results[index][-1]
    prediction = results[index][-2]
    im = Image.fromarray(results[index][:-2].reshape(28,28))
    im.convert('RGB').save("../lib/images/i%s_p%s_a%s.png"%(index,prediction, answer),"PNG")

