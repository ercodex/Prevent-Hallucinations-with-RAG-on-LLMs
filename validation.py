from lxml import etree

# Path of files
xml_file = "knowledge_base.xml"
dtd_file = "knowledge_base.dtd"

# Load DTD
with open(dtd_file, 'r') as dtd_f:
    dtd = etree.DTD(dtd_f)

# Open XML file and validate
with open(xml_file, 'r') as xml_f:
    xml_content = xml_f.read()

if dtd.validate(etree.XML(xml_content)): # If valid
    print("XML is valid!")
else:
    print("XML is not valid!") # If not valid
    print(dtd.error_log)

