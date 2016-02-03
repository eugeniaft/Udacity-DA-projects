## OpenStreetMap Project
## Data Wrangling with MongoDB
### Eugenia Fernandez
                          
### Map Area: Syracuse, NY, United States
                                                
https://www.openstreetmap.org/relation/174916

#### Initial Exploration of the data
After downloading the OSM file, I parsed through the file using ElementTree to find the number of each type of element and checked the k values for each tag to see if there were any problematic characters 
```python
def count_tags(filename):
        parser = ET.iterparse(filename)
        tags = {}
        count = 1
        for event, elem in parser:
            if elem.tag not in tags.keys():
                tags[elem.tag] = count
            else:
                tags[elem.tag] += count       
        return tags    
  ```      

{'node': 268949, 'nd': 316678, 'bounds': 1, 'member': 4997, 'tag': 195506, 'relation': 689, 'way': 34010, 'osm': 1}

```python
lower = re.compile(r'^([a-z]|_)*$')
lower_colon = re.compile(r'^([a-z]|_)*:([a-z]|_)*$')
problemchars = re.compile(r'[=\+/&<>;\'"\?%#$@\,\. \t\r\n]')

def key_type(element, keys):
    count = 1
    if element.tag == "tag":
        l = element.get('k')
        if lower.search(l):
            keys['lower'] += count
        elif lower_colon.search(l):
            keys['lower_colon'] += count
        elif problemchars.search(l):
            keys['problemchars'] += count
        else:
            keys['other'] += count
    return keys    
    

```

{'problemchars': 0, 'lower': 92944, 'other': 6583, 'lower_colon': 95979}

It appears that there are no problematic characters in the keys so I will not worry about that.

###1. Problems Encountered in the Map
After an initial audit of the data I found the following inconsistencies:

- Inconsistent or incorrect postal codes: “132179211”, “13219-331”, “14224”
- Inconsistent phone number formats: “+1 315 4354670”, “315 448 8005”, “001 315 857 0079”
- Inconsistent cuisine names or containing non-alphanumeric characters: “coffee_shop”, “donut”, “Fish”

I also checked for over-abbreviated street names but found no problems with that.

####Postal Codes

Auditing the data for possible issues in postal codes led me to discover that not all of them followed the same format, i.e., some had five-digits and some included a four-digit extension. Also, some postcodes presented an incorrect code, for instance, “14224” which corresponds to the city of Buffalo. 
In order to use most of the data, I decided to use only the five-digit codes so I defined a Python function to strip the four-digit extension from all zip codes that included one and dropped the incorrect codes from the data. 

```python
def new_postal(postcode):
   if re.match(r'^13[0-9]{3}$', postcode):
      continue
   else:
      ps_re = re.compile(r'(\d{5})')
      ps = ps_re.search(postcode)
      if ps:                        
          postals = ps.groups()
          new_postalcode = postals[0]
          if new_postalcode.startswith("14"):
              new_postalcode = None
          else:
              new_postalcode = None
          print postcode, "=>",  new_postalcode 
```

Example output:
<p>132179211 => 1321
<p>13202-1107 => 13202
<p>13219-331 => 13219
<p>14224 => None

####Phone numbers

When auditing phone numbers, I encountered that phone entries did not follow the same format. I decided to follow a 10-digit format of the form XXX-XXX-XXXX. Therefore, I defined a Python function to remove all characters before the area code, remove parenthesis, and include a hyphen instead of a space. 

```python
def new_phone_format(phone_num):
    if re.match(r'^[0-9]{3}-[0-9]{3}-[0-9]{4}$', phone_num):                
        new_phone = phone_num
    else:
        p_re = re.compile(r'(\d{3})\D*(\d{3})\D*(\d{4})$')
        ph = p_re.search(phone_num)
        if ph:                        
        phones = ph.groups()
        new_phone = phones[0] + "-" + phones[1] + "-" + phones[2]
    print phone_num, "=>", new_phone
```

Example output:

<p>+1-315-446-3200 => 315-446-3200
<p>315 437 3394 => 315-437-3394
<p>+1 315 652 4242 => 315-652-4242
<p>001 315 857 0079 => 315-857-0079


####Cuisine Names

When auditing cuisine names I found that not all of them followed the same convention. For example some restaurants serving seafood had for cuisine “seafood” while one had “fish”. Also, there were some restaurants with overly specific cuisine labels, for instance some coffee shops had “donut” or “bagel” as a cuisine, so I decided to group these restaurants with a broader label, in this case “coffee shops”.  Lastly, I capitalized all cuisine names. 

```python
cuisine_expected = ["american", "chinese", "italian", "french", "japanese", 
                    "sushi", "barbaque", "indian", "thai", "pizza", 
                    "burger", "mexican", "middle eastern", "coffee shop",
                    "diner", "polish", "sandwich", "seafood"] 

# UPDATE THIS VARIABLE
mapping = { "ice_cream": "ice cream",
            "coffee_shop": "coffee shop",
            "Middle_Eastern" : "middle eastern",
            "fish" : "seafood",
            "donut" : "coffee shop",
            "bagel" : "coffee shop"
            }

def new_cuisine(old_cuisine):
    cuisine_name = ""
    if old_cuisine in mapping.keys():
        cuisine_name = mapping[old_cuisine]
    else:
        cuisine_name = old_cuisine
    better_cuisine_name = cuisine_name.title()
    print old_cuisine, "=>", better_cuisine_name
```

Example output:

<p>fish => Seafood
<p>Pizza => Pizza
<p>donut => Coffee Shop
<p>coffee_shop => Coffee Shop
<p>italian => Italian

###2. Overview of the Data
In this section, basic statistics about the dataset using MongoDB queries are presented.                                                
                                                
File sizes
                                                
<p>syracuse.osm ......... 58.4 MB
<p>syracuse.osm.json .... 65.8 MB
                                                
```python
#Number of Documents
db.maps.find().count()
```
1514795

```python
#Number of Nodes
db.maps.find({"type":"node"}).count()
```
1344735

```python
#Number of Ways
db.maps.find({"type":"way"}).count()
```
169965

```python
#Number of Unique Users
len(db.maps.distinct('created.user'))
```
201
                                       
```python
#Top Contributing User
db.maps.aggregate([{"$group":{"_id":"$created.user", "count":{"$sum":1}}}, 
                   {"$sort":{"count":-1}}, 
                   {"$limit":1}])
```
[{u'count': 459999, u'_id': u'zeromap'}]

```python
#Number of Users appearing only once
db.maps.aggregate([{"$group":{"_id":"$created.user", "count":{"$sum":1}}}, 
                   {"$group":{"_id":"$count", "num_users":{"$sum":1}}}, 
                   {"$sort":{"_id":1}}, {"$limit":1}])
```
[{u'num_users': 43, u'_id': 3}]
```python
#Number of Documents Containing phone numbers
db.maps.find({"phone" : {"$exists" : 1}}).count()
```
2820
```python
#Number of Documents Containing postal codes
db.maps.find({"address.postcode" : {"$exists" : 1}}).count()
```
10128
```python
#Number of Documents Containing cuisine information
db.maps.find({"cuisine" : {"$exists" : 1}}).count()       
```
975

###3. Additional Ideas About the Data                                                       
For a user generated data, I found the data for the city of Syracuse fairly clean. It could benefit from having more information. For instance, not all restaurant entries have information about the opening hours, telephone number, website or price range. This type of information is useful for people when looking for places to eat. In order to motivate contribution by users, a gamification mechanism could be implemented were users get rewarded for their contributions in the way of badges or rankings. Some potential issues of implementing a gamification mechanism is to think of a good design where user content is checked since earning badges and rankings could turn into a pervasive incentive resulting in users contributing incorrect content or poorly researched information just to obtain rewards. A possible solution would be to have users be rewarded for checking the accuracy of other users’ contributions, that way there would be an incentive to contribute relevant and clean data. 
Also, regarding information for cuisine type, I noticed that it could be more informative to have two cuisine level information, one more broad cuisine type and other with more specific information about the specialty of the restaurant. For instance, for a bagel place, the level 1 cuisine type could be coffee shop and the level 2 could be bagels. This information could be taken out from values within the same node.
Additional data exploration using MongoDB queries                                             
````python
#Top 10 Amenities
db.maps.aggregate([{"$match" : {"amenity" : {"$exists" : 1}}}, 
                   {"$group" : {"_id" : "$amenity", "count" : {"$sum" : 1}}}, 
                   {"$sort" : {"count" : -1}}, 
                   {"$limit" : 10}])
                   
```
[{u'count': 3528, u'_id': u'parking'}, 
{u'count': 772, u'_id': u'school'}, 
{u'count': 544, u'_id': u'bench'}, 
{u'count': 512, u'_id': u'restaurant'}, 
{u'count': 472, u'_id': u'place_of_worship'}, 
{u'count': 464, u'_id': u'fast_food'}, 
{u'count': 412, u'_id': u'fuel'}, 
{u'count': 240, u'_id': u'bank'}, 
{u'count': 204, u'_id': u'post_box'}, 
{u'count': 176, u'_id': u'pharmacy'}]
 
 
```python
# Most popular cuisines
db.maps.aggregate([{"$match":{"amenity":{"$exists":1},
                  "amenity":"restaurant"}},
                   {"$group":{"_id":"$cuisine", "count":{"$sum":1}}}, 
                   {"$sort":{"count":-1}}, 
                   {"$limit":10}])
```
output not including incomplete or None types                   
[{u'count': 51, u'_id': u'Pizza'}, 
{u'count': 34, u'_id': u'American'}, 
{u'count': 29, u'_id': u'Italian'}, 
{u'count': 21, u'_id': u'Chinese'}, 
{u'count': 15, u'_id': u'Indian'}, 
{u'count': 15, u'_id': u'Mexican'}, 
{u'count': 11, u'_id': u'Seafood'}, 
{u'count': 11, u'_id': u'Japanese'}]
```python
# Biggest religion           
db.maps.aggregate([{"$match":{"amenity":{"$exists":1},
        "amenity":"place_of_worship"}}, 
                   {"$group":{"_id":"$religion", "count":{"$sum":1}}}, 
                   {"$sort":{"count":-1}}, 
                   {"$limit":1}])
                   
```
[{u'count': 424, u'_id': u'christian'}]
```python
#Top 10 Leisure Activities
db.maps.aggregate([{"$match":{"leisure":{"$exists":1}}}, 
                   {"$group":{"_id":"$leisure", "count":{"$sum":1}}},
                   {"$sort":{"count":-1}}, 
                   {"$limit":10}])
                   
```
[{u'count': 2420, u'_id': u'pitch'}, 
{u'count': 508, u'_id': u'park'}, 
{u'count': 344, u'_id': u'playground'}, 
{u'count': 116, u'_id': u'golf_course'}, 
{u'count': 100, u'_id': u'swimming_pool'}, 
{u'count': 64, u'_id': u'stadium'}, 
{u'count': 48, u'_id': u'sports_centre'}, 
{u'count': 36, u'_id': u'track'}, 
{u'count': 36, u'_id': u'picnic_table'}, 
{u'count': 28, u'_id': u'garden'}]                                                

###Conclusion    
After looking at the data for the city of Syracuse, I found the dataset to be fairly cleaned, although it was not complete. Several information about phone numbers, postcodes, among other things where missing. The major problems with the dataset where regarding inconsistencies in formats and some incorrect information. I was not surprised to see that the most popular cuisine was pizza followed by American, but I was surprised that the number one amenity was parking and the number three was bench, I thought it would have been restaurants and libraries since a lot of students live there.


