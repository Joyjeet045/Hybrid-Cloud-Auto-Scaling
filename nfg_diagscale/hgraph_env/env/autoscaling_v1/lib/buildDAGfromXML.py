
import xml.etree.ElementTree as ET

import networkx as nx
from config.param import configs

scale = configs.scale

def buildGraph(type, filename):
    tot_processTime = 0
    dag = nx.DiGraph(type=type)
    with open(filename, 'rb') as xml_file:
        tree = ET.parse(xml_file)
        xml_file.close()
    root = tree.getroot()
    for child in root:
        if child.tag == '{http://pegasus.isi.edu/schema/DAX}job':
            dag.add_node(int(child.attrib['id'][2:]), processTime=float(child.attrib['runtime']) / scale) 
            tot_processTime += float(child.attrib['runtime'])
        if child.tag == '{http://pegasus.isi.edu/schema/DAX}child':
            kid = int(child.attrib['ref'][2:])
            for p in child:
                parent = int(p.attrib['ref'][2:])
                tot_processTime += float(child.attrib['comm']) / scale
                dag.add_edge(parent, kid, weight=float(child.attrib['comm']) / scale)
        
    return dag, tot_processTime





