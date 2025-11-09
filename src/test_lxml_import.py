import lxml
print('lxml file:', lxml.__file__)
try:
    from lxml import etree
    print('etree import successful')
except Exception as e:
    print('etree import failed:', e)
