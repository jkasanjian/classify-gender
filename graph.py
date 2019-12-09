import matplotlib.pyplot as plt



def main():
    output_ImClass, output_alexnet = load_output()

    t_l, t_a, v_l, v_a = extract_numbers(output_ImClass)
    graph(t_l, t_a, v_l, v_a, '6 Layer CNN')

    t_l, t_a, v_l, v_a = extract_numbers(output_alexnet)
    graph(t_l, t_a, v_l, v_a, 'Alexnet')




def graph(t_l, t_a, v_l, v_a, name):
    plt.plot(t_a)
    plt.plot(v_a)
    name_a = name + ' Accuracy'
    plt.title(name_a)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.xlim([1, 74])
    plt.show()
    # summarize history for loss
    plt.plot(t_l)
    plt.plot(v_l)
    name_l = name + ' Loss'
    plt.title(name_l)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.xlim([1, 74])
    plt.show()

    return 



def extract_numbers(output):
    t_l = []
    t_a = []
    v_l = []
    v_a = []    
    output += '12345'

    count = 1
    for i in range(len(output)):
        if output[i] == ':':
            if count == 1:
                t_l.append(float(output[i+2 : i+8]))
                count += 1
            elif count == 2:
                t_a.append(float(output[i+2 : i+8]))
                count += 1
            elif count == 3:
                v_l.append(float(output[i+2 : i+8]))
                count += 1
            elif count == 4:
                v_a.append(float(output[i+2 : i+8]))
                count = 1

    return t_l, t_a, v_l, v_a
            
    

   
def load_output():
    output_ImClass = '''19/19 [==============================] - 68s 4s/step - loss: 0.7953 - accuracy: 0.5713 - val_loss: 0.6129 - val_accuracy: 0.6953
    Epoch 2/74
    19/19 [==============================] - 58s 3s/step - loss: 0.6200 - accuracy: 0.6819 - val_loss: 0.5200 - val_accuracy: 0.7695
    Epoch 3/74
    19/19 [==============================] - 63s 3s/step - loss: 0.5503 - accuracy: 0.7366 - val_loss: 0.4860 - val_accuracy: 0.7773
    Epoch 4/74
    19/19 [==============================] - 63s 3s/step - loss: 0.5356 - accuracy: 0.7395 - val_loss: 0.4729 - val_accuracy: 0.8047
    Epoch 5/74
    19/19 [==============================] - 62s 3s/step - loss: 0.4871 - accuracy: 0.7753 - val_loss: 0.4676 - val_accuracy: 0.7812
    Epoch 6/74
    19/19 [==============================] - 58s 3s/step - loss: 0.4843 - accuracy: 0.7860 - val_loss: 0.4029 - val_accuracy: 0.8281
    Epoch 7/74
    19/19 [==============================] - 62s 3s/step - loss: 0.4006 - accuracy: 0.8281 - val_loss: 0.3228 - val_accuracy: 0.8750
    Epoch 8/74
    19/19 [==============================] - 61s 3s/step - loss: 0.3664 - accuracy: 0.8458 - val_loss: 0.4077 - val_accuracy: 0.8125
    Epoch 9/74
    19/19 [==============================] - 60s 3s/step - loss: 0.3430 - accuracy: 0.8631 - val_loss: 0.2705 - val_accuracy: 0.8984
    Epoch 10/74
    19/19 [==============================] - 65s 3s/step - loss: 0.3481 - accuracy: 0.8561 - val_loss: 0.2841 - val_accuracy: 0.8789
    Epoch 11/74
    19/19 [==============================] - 61s 3s/step - loss: 0.2834 - accuracy: 0.8869 - val_loss: 0.2504 - val_accuracy: 0.8945
    Epoch 12/74
    19/19 [==============================] - 61s 3s/step - loss: 0.3030 - accuracy: 0.8769 - val_loss: 0.2906 - val_accuracy: 0.8867
    Epoch 13/74
    19/19 [==============================] - 61s 3s/step - loss: 0.2721 - accuracy: 0.8937 - val_loss: 0.2351 - val_accuracy: 0.9023
    Epoch 14/74
    19/19 [==============================] - 58s 3s/step - loss: 0.2445 - accuracy: 0.9032 - val_loss: 0.3183 - val_accuracy: 0.8633
    Epoch 15/74
    19/19 [==============================] - 60s 3s/step - loss: 0.2493 - accuracy: 0.8993 - val_loss: 0.2395 - val_accuracy: 0.8906
    Epoch 16/74
    19/19 [==============================] - 61s 3s/step - loss: 0.2322 - accuracy: 0.9054 - val_loss: 0.1973 - val_accuracy: 0.9258
    Epoch 17/74
    19/19 [==============================] - 62s 3s/step - loss: 0.2154 - accuracy: 0.9100 - val_loss: 0.1989 - val_accuracy: 0.9141
    Epoch 18/74
    19/19 [==============================] - 61s 3s/step - loss: 0.2980 - accuracy: 0.8892 - val_loss: 0.2594 - val_accuracy: 0.9062
    Epoch 19/74
    19/19 [==============================] - 59s 3s/step - loss: 0.2001 - accuracy: 0.9190 - val_loss: 0.2137 - val_accuracy: 0.9141
    Epoch 20/74
    19/19 [==============================] - 61s 3s/step - loss: 0.1833 - accuracy: 0.9252 - val_loss: 0.1835 - val_accuracy: 0.9336
    Epoch 21/74
    19/19 [==============================] - 62s 3s/step - loss: 0.2132 - accuracy: 0.9137 - val_loss: 0.1971 - val_accuracy: 0.9258
    Epoch 22/74
    19/19 [==============================] - 58s 3s/step - loss: 0.1738 - accuracy: 0.9328 - val_loss: 0.1750 - val_accuracy: 0.9336
    Epoch 23/74
    19/19 [==============================] - 61s 3s/step - loss: 0.2002 - accuracy: 0.9254 - val_loss: 0.1832 - val_accuracy: 0.9414
    Epoch 24/74
    19/19 [==============================] - 61s 3s/step - loss: 0.1773 - accuracy: 0.9293 - val_loss: 0.1897 - val_accuracy: 0.9180
    Epoch 25/74
    19/19 [==============================] - 61s 3s/step - loss: 0.1924 - accuracy: 0.9254 - val_loss: 0.1984 - val_accuracy: 0.9102
    Epoch 26/74
    19/19 [==============================] - 62s 3s/step - loss: 0.1707 - accuracy: 0.9367 - val_loss: 0.1962 - val_accuracy: 0.9180
    Epoch 27/74
    19/19 [==============================] - 61s 3s/step - loss: 0.1578 - accuracy: 0.9377 - val_loss: 0.1793 - val_accuracy: 0.9492
    Epoch 28/74
    19/19 [==============================] - 62s 3s/step - loss: 0.1760 - accuracy: 0.9307 - val_loss: 0.1529 - val_accuracy: 0.9375
    Epoch 29/74
    19/19 [==============================] - 59s 3s/step - loss: 0.1760 - accuracy: 0.9321 - val_loss: 0.1541 - val_accuracy: 0.9492
    Epoch 30/74
    19/19 [==============================] - 58s 3s/step - loss: 0.1401 - accuracy: 0.9427 - val_loss: 0.1653 - val_accuracy: 0.9336
    Epoch 31/74
    19/19 [==============================] - 61s 3s/step - loss: 0.1390 - accuracy: 0.9478 - val_loss: 0.1677 - val_accuracy: 0.9453
    Epoch 32/74
    19/19 [==============================] - 61s 3s/step - loss: 0.1444 - accuracy: 0.9465 - val_loss: 0.2221 - val_accuracy: 0.9062
    Epoch 33/74
    19/19 [==============================] - 61s 3s/step - loss: 0.1676 - accuracy: 0.9365 - val_loss: 0.1365 - val_accuracy: 0.9531
    Epoch 34/74
    19/19 [==============================] - 59s 3s/step - loss: 0.1439 - accuracy: 0.9461 - val_loss: 0.1422 - val_accuracy: 0.9336
    Epoch 35/74
    19/19 [==============================] - 61s 3s/step - loss: 0.1425 - accuracy: 0.9431 - val_loss: 0.1238 - val_accuracy: 0.9531
    Epoch 36/74
    19/19 [==============================] - 61s 3s/step - loss: 0.1413 - accuracy: 0.9443 - val_loss: 0.1276 - val_accuracy: 0.9414
    Epoch 37/74
    19/19 [==============================] - 61s 3s/step - loss: 0.1477 - accuracy: 0.9408 - val_loss: 0.1714 - val_accuracy: 0.9375
    Epoch 38/74
    19/19 [==============================] - 61s 3s/step - loss: 0.1232 - accuracy: 0.9535 - val_loss: 0.1410 - val_accuracy: 0.9414
    Epoch 39/74
    19/19 [==============================] - 58s 3s/step - loss: 0.1394 - accuracy: 0.9457 - val_loss: 0.1353 - val_accuracy: 0.9453
    Epoch 40/74
    19/19 [==============================] - 60s 3s/step - loss: 0.1268 - accuracy: 0.9511 - val_loss: 0.1097 - val_accuracy: 0.9375
    Epoch 41/74
    19/19 [==============================] - 61s 3s/step - loss: 0.1430 - accuracy: 0.9435 - val_loss: 0.1093 - val_accuracy: 0.9570
    Epoch 42/74
    19/19 [==============================] - 61s 3s/step - loss: 0.1321 - accuracy: 0.9517 - val_loss: 0.1192 - val_accuracy: 0.9414
    Epoch 43/74
    19/19 [==============================] - 60s 3s/step - loss: 0.1407 - accuracy: 0.9492 - val_loss: 0.1083 - val_accuracy: 0.9570
    Epoch 44/74
    19/19 [==============================] - 62s 3s/step - loss: 0.1199 - accuracy: 0.9548 - val_loss: 0.1015 - val_accuracy: 0.9570
    Epoch 45/74
    19/19 [==============================] - 58s 3s/step - loss: 0.1361 - accuracy: 0.9504 - val_loss: 0.1217 - val_accuracy: 0.9492
    Epoch 46/74
    19/19 [==============================] - 61s 3s/step - loss: 0.1389 - accuracy: 0.9490 - val_loss: 0.1013 - val_accuracy: 0.9531
    Epoch 47/74
    19/19 [==============================] - 61s 3s/step - loss: 0.1180 - accuracy: 0.9542 - val_loss: 0.1027 - val_accuracy: 0.9492
    Epoch 48/74
    19/19 [==============================] - 58s 3s/step - loss: 0.1351 - accuracy: 0.9532 - val_loss: 0.1233 - val_accuracy: 0.9453
    Epoch 49/74
    19/19 [==============================] - 60s 3s/step - loss: 0.1191 - accuracy: 0.9548 - val_loss: 0.1033 - val_accuracy: 0.9648
    Epoch 50/74
    19/19 [==============================] - 61s 3s/step - loss: 0.1333 - accuracy: 0.9525 - val_loss: 0.1113 - val_accuracy: 0.9531
    Epoch 51/74
    19/19 [==============================] - 61s 3s/step - loss: 0.1172 - accuracy: 0.9564 - val_loss: 0.0920 - val_accuracy: 0.9570
    Epoch 52/74
    19/19 [==============================] - 61s 3s/step - loss: 0.1191 - accuracy: 0.9574 - val_loss: 0.0924 - val_accuracy: 0.9609
    Epoch 53/74
    19/19 [==============================] - 61s 3s/step - loss: 0.1268 - accuracy: 0.9523 - val_loss: 0.1104 - val_accuracy: 0.9609
    Epoch 54/74
    19/19 [==============================] - 58s 3s/step - loss: 0.1200 - accuracy: 0.9580 - val_loss: 0.1043 - val_accuracy: 0.9453
    Epoch 55/74
    19/19 [==============================] - 58s 3s/step - loss: 0.1301 - accuracy: 0.9504 - val_loss: 0.0985 - val_accuracy: 0.9609
    Epoch 56/74
    19/19 [==============================] - 60s 3s/step - loss: 0.0945 - accuracy: 0.9659 - val_loss: 0.1488 - val_accuracy: 0.9414
    Epoch 57/74
    19/19 [==============================] - 61s 3s/step - loss: 0.1228 - accuracy: 0.9542 - val_loss: 0.1094 - val_accuracy: 0.9531
    Epoch 58/74
    19/19 [==============================] - 61s 3s/step - loss: 0.1112 - accuracy: 0.9601 - val_loss: 0.1079 - val_accuracy: 0.9414
    Epoch 59/74
    19/19 [==============================] - 61s 3s/step - loss: 0.1094 - accuracy: 0.9591 - val_loss: 0.1125 - val_accuracy: 0.9609
    Epoch 60/74
    19/19 [==============================] - 61s 3s/step - loss: 0.0995 - accuracy: 0.9624 - val_loss: 0.1209 - val_accuracy: 0.9531
    Epoch 61/74
    19/19 [==============================] - 61s 3s/step - loss: 0.1117 - accuracy: 0.9570 - val_loss: 0.1073 - val_accuracy: 0.9727
    Epoch 62/74
    19/19 [==============================] - 61s 3s/step - loss: 0.1112 - accuracy: 0.9585 - val_loss: 0.1008 - val_accuracy: 0.9531
    Epoch 63/74
    19/19 [==============================] - 59s 3s/step - loss: 0.1052 - accuracy: 0.9565 - val_loss: 0.1067 - val_accuracy: 0.9688
    Epoch 64/74
    19/19 [==============================] - 59s 3s/step - loss: 0.1277 - accuracy: 0.9485 - val_loss: 0.1087 - val_accuracy: 0.9531
    Epoch 65/74
    19/19 [==============================] - 61s 3s/step - loss: 0.0927 - accuracy: 0.9669 - val_loss: 0.0785 - val_accuracy: 0.9648
    Epoch 66/74
    19/19 [==============================] - 61s 3s/step - loss: 0.1050 - accuracy: 0.9644 - val_loss: 0.0870 - val_accuracy: 0.9727
    Epoch 67/74
    19/19 [==============================] - 60s 3s/step - loss: 0.1077 - accuracy: 0.9593 - val_loss: 0.1956 - val_accuracy: 0.9141
    Epoch 68/74
    19/19 [==============================] - 58s 3s/step - loss: 0.1083 - accuracy: 0.9560 - val_loss: 0.0979 - val_accuracy: 0.9609
    Epoch 69/74
    19/19 [==============================] - 61s 3s/step - loss: 0.1039 - accuracy: 0.9593 - val_loss: 0.1229 - val_accuracy: 0.9414
    Epoch 70/74
    19/19 [==============================] - 60s 3s/step - loss: 0.1048 - accuracy: 0.9607 - val_loss: 0.0926 - val_accuracy: 0.9727
    Epoch 71/74
    19/19 [==============================] - 60s 3s/step - loss: 0.0943 - accuracy: 0.9663 - val_loss: 0.1051 - val_accuracy: 0.9570
    Epoch 72/74
    19/19 [==============================] - 61s 3s/step - loss: 0.1016 - accuracy: 0.9616 - val_loss: 0.1142 - val_accuracy: 0.9570
    Epoch 73/74
    19/19 [==============================] - 60s 3s/step - loss: 0.1068 - accuracy: 0.9591 - val_loss: 0.0824 - val_accuracy: 0.9609
    Epoch 74/74
    19/19 [==============================] - 60s 3s/step - loss: 0.0877 - accuracy: 0.9685 - val_loss: 0.1053 - val_accuracy: 0.9609'''

    output_alexnet = '''19/19 [==============================] - 147s 8s/step - loss: 0.8067 - accuracy: 0.6343 - val_loss: 0.9348 - val_accuracy: 0.5391
    Epoch 3/75
    19/19 [==============================] - 149s 8s/step - loss: 0.6682 - accuracy: 0.6639 - val_loss: 0.6279 - val_accuracy: 0.6484
    Epoch 4/75
    19/19 [==============================] - 139s 7s/step - loss: 0.6093 - accuracy: 0.7050 - val_loss: 0.5773 - val_accuracy: 0.7656
    Epoch 5/75
    19/19 [==============================] - 84s 4s/step - loss: 0.5043 - accuracy: 0.7732 - val_loss: 0.5980 - val_accuracy: 0.6328
    Epoch 6/75
    19/19 [==============================] - 70s 4s/step - loss: 0.4290 - accuracy: 0.8062 - val_loss: 0.6336 - val_accuracy: 0.6406
    Epoch 7/75
    19/19 [==============================] - 72s 4s/step - loss: 0.3510 - accuracy: 0.8499 - val_loss: 0.4573 - val_accuracy: 0.7695
    Epoch 8/75
    19/19 [==============================] - 73s 4s/step - loss: 0.2887 - accuracy: 0.8775 - val_loss: 0.2501 - val_accuracy: 0.8906
    Epoch 9/75
    19/19 [==============================] - 72s 4s/step - loss: 0.2483 - accuracy: 0.8970 - val_loss: 0.2081 - val_accuracy: 0.9062
    Epoch 10/75
    19/19 [==============================] - 72s 4s/step - loss: 0.2473 - accuracy: 0.8962 - val_loss: 0.2293 - val_accuracy: 0.9141
    Epoch 11/75
    19/19 [==============================] - 71s 4s/step - loss: 0.2374 - accuracy: 0.8988 - val_loss: 0.2535 - val_accuracy: 0.9062
    Epoch 12/75
    19/19 [==============================] - 68s 4s/step - loss: 0.2186 - accuracy: 0.9129 - val_loss: 0.2416 - val_accuracy: 0.8984
    Epoch 13/75
    19/19 [==============================] - 71s 4s/step - loss: 0.2060 - accuracy: 0.9147 - val_loss: 0.2323 - val_accuracy: 0.9023
    Epoch 14/75
    19/19 [==============================] - 68s 4s/step - loss: 0.2244 - accuracy: 0.9226 - val_loss: 0.2526 - val_accuracy: 0.8984
    Epoch 15/75
    19/19 [==============================] - 72s 4s/step - loss: 0.2192 - accuracy: 0.9120 - val_loss: 0.5124 - val_accuracy: 0.7930
    Epoch 16/75
    19/19 [==============================] - 72s 4s/step - loss: 0.2010 - accuracy: 0.9178 - val_loss: 0.2281 - val_accuracy: 0.8945
    Epoch 17/75
    19/19 [==============================] - 72s 4s/step - loss: 0.2011 - accuracy: 0.9161 - val_loss: 0.3562 - val_accuracy: 0.8516
    Epoch 18/75
    19/19 [==============================] - 69s 4s/step - loss: 0.1991 - accuracy: 0.9263 - val_loss: 0.2492 - val_accuracy: 0.8984
    Epoch 19/75
    19/19 [==============================] - 72s 4s/step - loss: 0.1798 - accuracy: 0.9289 - val_loss: 0.1886 - val_accuracy: 0.9141
    Epoch 20/75
    19/19 [==============================] - 72s 4s/step - loss: 0.1628 - accuracy: 0.9377 - val_loss: 0.2292 - val_accuracy: 0.9044
    Epoch 21/75
    19/19 [==============================] - 76s 4s/step - loss: 0.1714 - accuracy: 0.9347 - val_loss: 0.2309 - val_accuracy: 0.9219
    Epoch 22/75
    19/19 [==============================] - 73s 4s/step - loss: 0.1625 - accuracy: 0.9369 - val_loss: 0.1610 - val_accuracy: 0.9492
    Epoch 23/75
    19/19 [==============================] - 72s 4s/step - loss: 0.1579 - accuracy: 0.9389 - val_loss: 0.1689 - val_accuracy: 0.9336
    Epoch 24/75
    19/19 [==============================] - 73s 4s/step - loss: 0.1516 - accuracy: 0.9435 - val_loss: 0.1406 - val_accuracy: 0.9492
    Epoch 25/75
    19/19 [==============================] - 73s 4s/step - loss: 0.1397 - accuracy: 0.9445 - val_loss: 0.1580 - val_accuracy: 0.9219
    Epoch 26/75
    19/19 [==============================] - 72s 4s/step - loss: 0.1486 - accuracy: 0.9369 - val_loss: 0.3808 - val_accuracy: 0.8242
    Epoch 27/75
    19/19 [==============================] - 69s 4s/step - loss: 0.1574 - accuracy: 0.9429 - val_loss: 0.1599 - val_accuracy: 0.9414
    Epoch 28/75
    19/19 [==============================] - 73s 4s/step - loss: 0.1457 - accuracy: 0.9451 - val_loss: 0.1812 - val_accuracy: 0.9219
    Epoch 29/75
    19/19 [==============================] - 72s 4s/step - loss: 0.1374 - accuracy: 0.9468 - val_loss: 0.1312 - val_accuracy: 0.9414
    Epoch 30/75
    19/19 [==============================] - 71s 4s/step - loss: 0.1283 - accuracy: 0.9513 - val_loss: 0.2082 - val_accuracy: 0.9219
    Epoch 31/75
    19/19 [==============================] - 71s 4s/step - loss: 0.1250 - accuracy: 0.9496 - val_loss: 0.2086 - val_accuracy: 0.9180
    Epoch 32/75
    19/19 [==============================] - 71s 4s/step - loss: 0.1476 - accuracy: 0.9414 - val_loss: 0.1594 - val_accuracy: 0.9258
    Epoch 33/75
    19/19 [==============================] - 68s 4s/step - loss: 0.1379 - accuracy: 0.9474 - val_loss: 0.1999 - val_accuracy: 0.9219
    Epoch 34/75
    19/19 [==============================] - 70s 4s/step - loss: 0.1287 - accuracy: 0.9515 - val_loss: 0.1273 - val_accuracy: 0.9375
    Epoch 35/75
    19/19 [==============================] - 71s 4s/step - loss: 0.1280 - accuracy: 0.9513 - val_loss: 0.2107 - val_accuracy: 0.9219
    Epoch 36/75
    19/19 [==============================] - 68s 4s/step - loss: 0.1327 - accuracy: 0.9485 - val_loss: 0.1882 - val_accuracy: 0.9258
    Epoch 37/75
    19/19 [==============================] - 72s 4s/step - loss: 0.1415 - accuracy: 0.9455 - val_loss: 0.1305 - val_accuracy: 0.9570
    Epoch 38/75
    19/19 [==============================] - 72s 4s/step - loss: 0.1207 - accuracy: 0.9542 - val_loss: 0.1922 - val_accuracy: 0.9453
    Epoch 39/75
    19/19 [==============================] - 71s 4s/step - loss: 0.1217 - accuracy: 0.9511 - val_loss: 0.1638 - val_accuracy: 0.9375
    Epoch 40/75
    19/19 [==============================] - 68s 4s/step - loss: 0.1327 - accuracy: 0.9519 - val_loss: 0.2638 - val_accuracy: 0.8529
    Epoch 41/75
    19/19 [==============================] - 79s 4s/step - loss: 0.1271 - accuracy: 0.9474 - val_loss: 0.1268 - val_accuracy: 0.9531
    Epoch 42/75
    19/19 [==============================] - 72s 4s/step - loss: 0.1123 - accuracy: 0.9583 - val_loss: 0.1381 - val_accuracy: 0.9375
    Epoch 43/75
    19/19 [==============================] - 71s 4s/step - loss: 0.1078 - accuracy: 0.9609 - val_loss: 0.2044 - val_accuracy: 0.9180
    Epoch 44/75
    19/19 [==============================] - 72s 4s/step - loss: 0.1181 - accuracy: 0.9548 - val_loss: 0.1977 - val_accuracy: 0.9297
    Epoch 45/75
    19/19 [==============================] - 68s 4s/step - loss: 0.1190 - accuracy: 0.9558 - val_loss: 0.1656 - val_accuracy: 0.9375
    Epoch 46/75
    19/19 [==============================] - 72s 4s/step - loss: 0.1245 - accuracy: 0.9529 - val_loss: 0.1505 - val_accuracy: 0.9336
    Epoch 47/75
    19/19 [==============================] - 68s 4s/step - loss: 0.1555 - accuracy: 0.9500 - val_loss: 0.1158 - val_accuracy: 0.9609
    Epoch 48/75
    19/19 [==============================] - 72s 4s/step - loss: 0.1145 - accuracy: 0.9554 - val_loss: 0.1149 - val_accuracy: 0.9531
    Epoch 49/75
    19/19 [==============================] - 72s 4s/step - loss: 0.0878 - accuracy: 0.9657 - val_loss: 0.1359 - val_accuracy: 0.9336
    Epoch 50/75
    19/19 [==============================] - 72s 4s/step - loss: 0.1137 - accuracy: 0.9572 - val_loss: 0.1025 - val_accuracy: 0.9688
    Epoch 51/75
    19/19 [==============================] - 71s 4s/step - loss: 0.1043 - accuracy: 0.9605 - val_loss: 0.1675 - val_accuracy: 0.9219
    Epoch 52/75
    19/19 [==============================] - 71s 4s/step - loss: 0.0993 - accuracy: 0.9626 - val_loss: 0.2417 - val_accuracy: 0.8945
    Epoch 53/75
    19/19 [==============================] - 71s 4s/step - loss: 0.1184 - accuracy: 0.9546 - val_loss: 0.1554 - val_accuracy: 0.9375
    Epoch 54/75
    19/19 [==============================] - 67s 4s/step - loss: 0.1235 - accuracy: 0.9595 - val_loss: 0.1642 - val_accuracy: 0.9297
    Epoch 55/75
    19/19 [==============================] - 68s 4s/step - loss: 0.1147 - accuracy: 0.9621 - val_loss: 0.1823 - val_accuracy: 0.9219
    Epoch 56/75
    19/19 [==============================] - 71s 4s/step - loss: 0.1688 - accuracy: 0.9363 - val_loss: 0.1830 - val_accuracy: 0.9336
    Epoch 57/75
    19/19 [==============================] - 70s 4s/step - loss: 0.1203 - accuracy: 0.9537 - val_loss: 0.1507 - val_accuracy: 0.9258
    Epoch 58/75
    19/19 [==============================] - 71s 4s/step - loss: 0.1089 - accuracy: 0.9593 - val_loss: 0.2079 - val_accuracy: 0.9219
    Epoch 59/75
    19/19 [==============================] - 71s 4s/step - loss: 0.0981 - accuracy: 0.9663 - val_loss: 0.1418 - val_accuracy: 0.9297
    Epoch 60/75
    19/19 [==============================] - 71s 4s/step - loss: 0.0843 - accuracy: 0.9677 - val_loss: 0.2159 - val_accuracy: 0.9412
    Epoch 61/75
    19/19 [==============================] - 75s 4s/step - loss: 0.1289 - accuracy: 0.9580 - val_loss: 0.1046 - val_accuracy: 0.9531
    Epoch 62/75
    19/19 [==============================] - 71s 4s/step - loss: 0.1000 - accuracy: 0.9640 - val_loss: 0.1542 - val_accuracy: 0.9375
    Epoch 63/75
    19/19 [==============================] - 71s 4s/step - loss: 0.0942 - accuracy: 0.9648 - val_loss: 0.1801 - val_accuracy: 0.9492
    Epoch 64/75
    19/19 [==============================] - 71s 4s/step - loss: 0.0816 - accuracy: 0.9714 - val_loss: 0.1490 - val_accuracy: 0.9453
    Epoch 65/75
    19/19 [==============================] - 71s 4s/step - loss: 0.0919 - accuracy: 0.9661 - val_loss: 0.1572 - val_accuracy: 0.9531
    Epoch 66/75
    19/19 [==============================] - 68s 4s/step - loss: 0.1310 - accuracy: 0.9593 - val_loss: 0.2273 - val_accuracy: 0.9023
    Epoch 67/75
    19/19 [==============================] - 72s 4s/step - loss: 0.1474 - accuracy: 0.9428 - val_loss: 0.2521 - val_accuracy: 0.9062
    Epoch 68/75
    19/19 [==============================] - 72s 4s/step - loss: 0.1294 - accuracy: 0.9486 - val_loss: 0.1876 - val_accuracy: 0.9258
    Epoch 69/75
    19/19 [==============================] - 68s 4s/step - loss: 0.1482 - accuracy: 0.9466 - val_loss: 0.1202 - val_accuracy: 0.9531
    Epoch 70/75
    19/19 [==============================] - 71s 4s/step - loss: 0.1267 - accuracy: 0.9523 - val_loss: 0.1469 - val_accuracy: 0.9453
    Epoch 71/75
    19/19 [==============================] - 71s 4s/step - loss: 0.1028 - accuracy: 0.9634 - val_loss: 0.1540 - val_accuracy: 0.9375
    Epoch 72/75
    19/19 [==============================] - 71s 4s/step - loss: 0.0988 - accuracy: 0.9655 - val_loss: 0.1747 - val_accuracy: 0.9219
    Epoch 73/75
    19/19 [==============================] - 67s 4s/step - loss: 0.0970 - accuracy: 0.9668 - val_loss: 0.2259 - val_accuracy: 0.9023
    Epoch 74/75
    19/19 [==============================] - 71s 4s/step - loss: 0.1143 - accuracy: 0.9581 - val_loss: 0.1051 - val_accuracy: 0.9648
    Epoch 75/75
    19/19 [==============================] - 71s 4s/step - loss: 0.0903 - accuracy: 0.9624 - val_loss: 0.0909 - val_accuracy: 0.9648
    '''

    return output_ImClass, output_alexnet

   




main()