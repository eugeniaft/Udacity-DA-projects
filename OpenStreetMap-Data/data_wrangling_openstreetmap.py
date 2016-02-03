# -*- coding: utf-8 -*-
"""
Created on Tue Aug 04 13:13:41 2015

@author: Eugenia
"""
import xml.etree.cElementTree as ET
import re
import codecs
import json

OSMFILE = "C:\Users\Eugenia\Desktop\syracuse.osm"
    
def new_phone_format(elem):
    new_phone = ""        
    if elem.tag == "node" or elem.tag == "way":                                       
        for tag in elem.iter("tag"):
            if tag.attrib['k'] == 'phone':
                if re.match(r'^[0-9]{3}-[0-9]{3}-[0-9]{4}$', tag.attrib['v']):                        
                    new_phone = tag.attrib['v']                        
                else:
                    p_re = re.compile(r'(\d{3})\D*(\d{3})\D*(\d{4})$')
                    ph = p_re.search(tag.attrib['v'])
                    if ph:                        
                        phones = ph.groups()
                        new_phone = phones[0] + "-" + phones[1] + "-" + phones[2]        
    return new_phone
    
def new_postal(elem):
    new_postalcode = ""
    if elem.tag == "node" or elem.tag == "way" :                                       
        for tag in elem.iter("tag"):
            if tag.attrib['k'] == 'addr:postcode':
                if re.match(r'^13[0-9]{3}$', tag.attrib['v']):
                    new_postalcode = tag.attrib['v']
                else:
                    ps_re = re.compile(r'(\d{5})')
                    ps = ps_re.search(tag.attrib['v'])
                    if ps:                        
                        postals = ps.groups()
                        new_postalcode = postals[0]
                        if new_postalcode.startswith("14"):
                            new_postalcode = None                                
    return new_postalcode 

cuisine_expected = ["american", "chinese", "italian", "french", "japanese", "sushi", "barbaque", "indian", "thai", "pizza", 
                    "burger", "mexican", "middle eastern", "cofee shop", "diner", "polish", "sandwich", "seafood"] 

# UPDATE THIS VARIABLE
mapping = { "ice_cream": "ice cream",
            "coffee_shop": "coffee shop",
            "Middle_Eastern" : "middle eastern",
            "fish" : "seafood",
            "donut" : "coffee shop",
            "bagel" : "coffee shop"
            }

def new_cuisine(elem):
    cuisine_name = ""
    better_cuisine_name = ""
    if elem.tag == "node" or elem.tag == "way" :                                       
        for tag in elem.iter("tag"):
            if tag.attrib['k'] == 'cuisine':
                if tag.attrib['v'] in mapping.keys():
                    cuisine_name = mapping[tag.attrib['v']]
                else:
                    cuisine_name = tag.attrib['v']
            
                better_cuisine_name = cuisine_name.title()
            
    return better_cuisine_name
                   
CREATED = [ "version", "changeset", "timestamp", "user", "uid"]

#Function to transform xml file into json format
def shape_element(elem):
    node = {}
    created = {}
    address = {}
    coords = ["lat", "lon"]
    node_refs = []
    if elem.tag == "node" or elem.tag == "way" :
        node['type'] = elem.tag
        
        for key in elem.attrib:
            if key in CREATED:
                created[key] = elem.attrib[key]                         
            elif key in coords:               
                node['pos'] = [float(elem.attrib[coords[0]]), float(elem.attrib[coords[1]])]
            else:
                node[key] = elem.attrib[key]
        
        #Iterate over tag children        
        for tag in elem.iter("tag"):
            if tag.attrib["k"].startswith("addr:"):
                new_key = re.split(r'^addr:', tag.attrib["k"])                                
                if new_key[1].startswith("street:"):
                    break
                elif new_key[1].startswith("postcode"):
                    address['postcode'] = new_postal(elem)
                else:    
                    address[new_key[1]] = tag.attrib["v"]
                                   
            elif tag.attrib['k'] == 'phone':
                node['phone'] = new_phone_format(elem)
            
            elif tag.attrib["k"] == 'cuisine':
                node['cuisine'] = new_cuisine(elem)
            else:  
                node[tag.attrib["k"]] = tag.attrib["v"]                               
        
        if len(address) != 0:
            node['address'] = address
        if len(created) != 0:    
            node['created'] = created        
               
        if node['type']  == 'way':   
            for n in elem.iter("nd"):
                node_refs.append(n.attrib['ref'])
                node["node_refs"] = node_refs
                                
        return node
    else:
        return None

def process_map(file_in, pretty = False):
    file_out = "{0}.json".format(file_in)
    data = []
    with codecs.open(file_out, "w") as fo:
        for _, elem in ET.iterparse(file_in):
            el = shape_element(elem)
            if el:
                data.append(el)
                if pretty:
                    fo.write(json.dumps(el, indent=2)+"\n")
                else:
                    fo.write(json.dumps(el) + "\n")
        elem.clear()            
   
    return data

data = process_map(OSMFILE, False)
#Importing data set to MongoDB
from pymongo import MongoClient
client = MongoClient("mongodb://localhost:27017")
db = client.dbcity #createdatabase
collection = db.syr # create a collection
db.maps.insert(data) 

#MongoDB Queries

#Number of Documents
print db.maps.find().count()
#Number of Unique Users
print len(db.maps.distinct('created.user'))
#Top Contributing User
results1 = db.maps.aggregate([{"$group" : {"_id" : "$created.user", "count" : {"$sum" : 1}}}, \
                           {"$sort" : {"count" : -1}}, \
                           {"$limit" : 1}])

results_list1 = [res for res in results1]                           
print results_list1

#Number of Users appearing only once
results2 = db.maps.aggregate([{"$group":{"_id":"$created.user", "count":{"$sum":1}}}, {"$group":{"_id":"$count", "num_users":{"$sum":1}}}, {"$sort":{"_id":1}}, {"$limit":1}])
results_list2 = [res for res in results2]
print results_list2
                                                      
#Number of Nodes
print db.maps.find({"type":"node"}).count()
#Number of Ways
print db.maps.find({"type":"way"}).count()
#Number of Documents Containing phone numbers
print db.maps.find({"phone" : {"$exists" : 1}}).count()
#Number of Documents Containing postal codes
print db.maps.find({"address.postcode" : {"$exists" : 1}}).count()
#Number of Documents Containing cuisine information
print db.maps.find({"cuisine" : {"$exists" : 1}}).count()       

# Biggest religion           
results3 = db.maps.aggregate([{"$match":{"amenity":{"$exists":1}, "amenity":"place_of_worship"}}, \
                           {"$group":{"_id":"$religion", "count":{"$sum":1}}}, \
                           {"$sort" : {"count" : -1}}, \
                           {"$limit":1}])

results_list3 = [res for res in results3]
print results_list3  
               
#Top 10 Amenities
results4 = db.maps.aggregate([{"$match" : {"amenity" : {"$exists" : 1}}}, \
                           {"$group" : {"_id" : "$amenity", "count" : {"$sum" : 1}}}, \
                           {"$sort" : {"count" : -1}}, \
                           {"$limit" : 10}])
                           
results_list4 = [res for res in results4]                               
print results_list4

# Most popular cuisines
results5 = db.maps.aggregate([{"$match":{"amenity":{"$exists":1}, "amenity":"restaurant"}}, \
                           {"$group":{"_id":"$cuisine", "count":{"$sum":1}}}, \
                           {"$sort" : {"count" : -1}}, \
                           {"$limit":10}])

results_list5 = [res for res in results5]
print results_list5                                             

#Top 10 Leisures
results6 = db.maps.aggregate([{"$match" : {"leisure" : {"$exists" : 1}}}, \
                           {"$group" : {"_id" : "$leisure", "count" : {"$sum" : 1}}}, \
                           {"$sort" : {"count" : -1}}, \
                           {"$limit" : 10}])

results_list6 = [res for res in results6]
print results_list6

                        
