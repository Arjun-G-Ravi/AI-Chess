with open('data/test.db', 'rb') as f:
    data = f.read().decode('utf-8')

    

print(len(data))
print(data[:1000])