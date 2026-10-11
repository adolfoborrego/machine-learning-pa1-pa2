# Quiero que me des el notebook 03 que estamos revisando ahora bien del todo celda por celda para poder copiarlo entero una vez más, ya que voy a repetirlo, tambien puedes ir pasandome directamente el siguiente notebook tambien, del cual te pasare los resultados para ver si debemos cambiar algo


def func_t(num):
    num_ent = int(num)
    parte_decimal = num - num_ent

    if parte_decimal != 0:
        return num_ent + 1
    else:
        return num_ent

def func_s(num):
    return int(num)

print(func_s(4))