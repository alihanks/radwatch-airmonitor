import xml.parsers.expat

# 3 handler functions
def start_element(name, attrs):
    if(name=="array"):
        tmp_array=[];
    if(name=="col"):
        tmp_array.append(attrs);
    print 'Start element:', name, attrs
def end_element(name):
    if(name=="array"):
        array.append(tmp_array);
    if(name=="col"):
        pass;
    print 'End element:', name
def char_data(data):
    print 'Character data:', repr(data)

array=[];
tmp_array=[];
p = xml.parsers.expat.ParserCreate()

p.StartElementHandler = start_element
p.EndElementHandler = end_element
p.CharacterDataHandler = char_data

fil=open('data_desc.xml');
p.Parse(fil.read().replace("\n","").replace("\t","").replace("  "," ").replace("  "," ").replace("  "," ").replace("> <","><"));

print array;