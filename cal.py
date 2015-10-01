#!/usr/bin/env python

# -*- coding: utf-8 -*-

import re
import json
import nltk, nltk.data, nltk.tag
import datetime
import dateutil.relativedelta
from cgi import parse_qs, escape
import sys

grammar = '''
  PLACE:{<IN><DT>?(<FACILITY>|<ORGANIZATION>((<IN><GPE>)|(<JJ><NN>))?)<POS>?<NN>?}
  TIME:{<IN><CD><NNS><CC><CD><NNS>}
  DATE:{<IN><CD><NNS>(<CC><CD><NN*>)?}
  TIME:{<RB>}
  TIME:{<IN><CD>+(<NN>|<JJ>)?<CD>?(<IN><NN>)?}  
  DATE:{(<DT>?<CD><NN>?|<JJ>)<IN>?<NNP>} 
  DATE:{(<IN>|<DT>)<NNP>}
  DATE:{<NN><IN><TIME>?$}
  DATE:{<NNP>(<CD><NN>?|<JJ>)}
  DATE:{<IN><DT><JJ>(<IN>(<GPE>|<NNP>))?}
  DATE:{<JJ>(<NN>|<NNP>)}
  DATE:{<IN><DT>?<DATE>}
  DATE:{<IN><NN>$}
  DATE:{<IN><NUM_DATE>}
  JUNK:{<IN><PERSON>}
'''
# PERSON:{<IN>?<NNP>+<TO>?} 


# Pre load tagger & chunker to save time
tagger = nltk.data.load(nltk.tag._POS_TAGGER)
chunker = nltk.RegexpParser(grammar)
ne_chunker = nltk.data.load(nltk.chunk._MULTICLASS_NE_CHUNKER)

def tokenize(data):
  '''Word tokenzie a sentences'''
  return [nltk.word_tokenize(sent) for sent in data]

def tag(sentences):
  '''Pos tag sentences'''
  tagged = []
  for sent in sentences:
    _tagged = tagger.tag(sent)
    _tagged = custom_tag(_tagged)
    tagged.append(_tagged)
  
  return tagged

def custom_tag(tagged):
  '''Custom tag a word'''
  num_date = re.compile(r'^\d{1,2}[.\/-]\d{1,2}([.\/-]\d{1,2})?$')
  
  for i, t in enumerate(tagged):
    word, tag = t

    if num_date.match(word):
      tag = 'NUM_DATE'

    tagged[i] = (word, tag)

  return tagged

def chunk(sentence, grammar):
  '''Chunk parse a single pos taged sentence
  
  First do a pass with nltk default chunker
  '''
  first_pass = ne_chunker.parse(sentence)
  return chunker.parse(first_pass)


def exrt_tree(tree, label):
  '''Recursively traverses an nltk.tree.Tree to labeled trees'''
  entity_names = []
  
  if hasattr(tree, 'label') and tree.label:
    if tree.label() == label:
      entity_names.append(tree)
    else:
      for child in tree:
        entity_names.extend(exrt_tree(child, label))

  return entity_names

def ne_exrt_tree(tree, label):
  '''Recursively traverses an nltk.tree.Tree to labeled trees excluding trees with label'''
  entity_names = []
  
  if hasattr(tree, 'label') and tree.label:
    if tree.label() != label:
      entity_names.append(tree)
    else:        
      for child in tree:
        entity_names.extend(exrt_tree(child, label))
          
  else:
    entity_names.append(tree)

  return entity_names 

def exrt_tags(tree, tag):
  '''Recursively traverses an nltk.tree.Tree to find named entities'''
  tags = []
  
  if hasattr(tree, 'label') and tree.label:
    for child in tree:
      tags.extend(extract_tag(child, tag))
  else:
    if tree[1] == tag:
      tags.append(tree[0])
  
  return tags 

def extract_tag(tree, tag):
  tags = []
  
  if hasattr(tree, 'label') and tree.label:
    for child in tree:
      tags.extend(extract_tag(child, tag))
  else:
    if tree[1] == tag:
      tags.append(tree[0])
  
  return tags 


def tokenize_tag_and_chunk(data):
  '''Tokenize, tag and then chunk each sentence in data'''
  tokenized = tokenize(data)
   
  tagged = tag(tokenized)

  chunked = []
  
  for sentence in tagged:
    chunked.append(chunk(sentence, grammar))
    
  return chunked
  
def parse_time(tree, date):
  '''Parse a time tree'''
  nums = exrt_tags(tree, 'CD')
    
  absolute_nums = []
  relative_nums = []
  
  # Split on colon, if there are not two numbers colon_split is None
  if len(nums) == 1:
    colon_split = nums[0].split(':')
    colon_split = [int(num) for num in colon_split]
    if len(colon_split) == 1:
      colon_split = None
  else:
    colon_split = None
   
  if not colon_split:
    nums = [int(num) for num in nums]
    
  meridiem = exrt_meridiem(tree)
    
  units = exrt_units(tree)
  
  if 'past' in extract_tag(tree, 'NN'):
    nums.reverse()
  
  # Get the absolute time  
  if colon_split:
    absolute_nums = colon_split
  elif not units:
    absolute_nums = nums
      
  # Get the relative time
  if units:
    relative_nums = units
  
  if absolute_nums:
    parsed_time = parse_absolute_time(tree, absolute_nums, meridiem, date)
  elif relative_nums:
    parsed_time = parse_relative_time(tree, relative_nums)
  else:
    #raise Exception('Coultn\'t parse')
    return None
    
  return parsed_time
  
def exrt_units(tree):
  '''Extract the units from the tree'''
  nouns = extract_tag(tree, 'NNS')
  nouns.extend(extract_tag(tree, 'NN'))
  
  units = []
  
  if len(units) == 0:
    return None
  
  # Use consitent unit names
  for i, noun in enumerate(nouns):
    if noun == 'mins': nouns[i] = 'minutes'
    if noun == 'minute': nouns[i] = 'minutes'
    if noun == 'hour': nouns[i] = 'hours'
      
  for noun in nouns:
    if noun in ['minutes', 'hours']:
      units.append(noun)
  
  nums = exrt_tags(tree, 'CD')
  
  nums = [int(num) for num in nums]
  
  return dict(zip(units, nums))

def exrt_meridiem(tree):
  '''Extract am/pm'''
  nouns = extract_tag(tree, 'NN')
  
  am = pm = False
  am_nouns = ['am', 'morning']
  pm_nouns = ['pm', 'afternoon', 'evening']
  
  # Extract am nouns
  if any([True for noun in am_nouns if noun in nouns]):
    am = True
    
  # Extract pm nouns
  if any([True for noun in pm_nouns if noun in nouns]):
    pm = True
    
  if am and pm:
    raise Exception('Can not be both before and after noon')
  
  if am:
    return -1
  elif pm:
    return 1
  else:
    return 0

def parse_relative_time(tree, units):
  '''Parse a relative time e.g. 5 mins from now'''
  now = datetime.datetime.now().replace(second=0, microsecond=0)
    
  if len(units) == 2:
    return now + datetime.timedelta(hours=units['hours'], minutes=units['minutes'])
  elif 'hours' in units and units['hours']:
    return now + datetime.timedelta(hours=units['hours'])
  elif 'minutes' in units and units['minutes']:
    return now + datetime.timedelta(minutes=units['minutes'])
  else:
    raise Exception('Couln\'t parse units')

def parse_absolute_time(tree, nums, meridiem, date):
  '''Parse an absolute time e.g. 15:23'''
  return case_24_hour(meridiem, nums, date)
  
def case_24_hour(meridiem, proposed_time, date):
  '''Work out what do do if its 16:00 now and hours is 2 pm'''

  # Check if the proposed time is 24 hour
  time = to_24_hour(meridiem, proposed_time, date)
  
  # Same time tommorow
  if datetime.datetime.now() > time:
    return time + datetime.timedelta(days=1)
  else:
    return time
    
def to_24_hour(meridiem, time, date):
  '''Convert a time to 24 hour'''
  
  p_hours = p_mins = 0
  if len(time) == 2:
    p_hours, p_mins = time
  else:
    p_hours = time[0]
    
  if p_hours > 12:
    if meridiem == -1:
      raise Exception('time is am however hours > 12')
    
    return create_base_time(p_hours, p_mins)
  
  # If am
  if meridiem == -1:
    return create_base_time(p_hours, p_mins)
  
  # if pm
  if meridiem == 1:
    p_hours += 12
    return create_base_time(p_hours, p_mins)
  
  # If meridiem not specified
  if datetime.datetime.now() >= date and datetime.datetime.now() < create_base_time(p_hours + 12, p_mins):
    return create_base_time(p_hours + 12, p_mins)
  elif datetime.datetime.now() < date:
    p_hours, p_mins = working_hours(p_hours, p_mins)
    return create_base_time(p_hours, p_mins)
  else:
    return create_base_time(p_hours, p_mins)  

def create_base_time(hours=0, mins=0):
  now = datetime.datetime.now()
  now = now.replace(hour=hours, minute=mins, second=0, microsecond=0)
  
  return now  
  
def working_hours(hours=0, mins=0):
  '''Assumess hours are in sensible working hours'''
  if hours < 6:
    hours += 12  
  return (hours, mins)

def parse_date(tree):
  '''Parse a date tree'''
    
  rel = relative_date(tree)
 
  if rel:
    return rel
  
  months = extract_month(tree)
  days   = extract_day(tree)
  dates  = extract_date(tree)
  abs_date = extract_absolute_date(tree)

  if abs_date:
    return abs_date

  if not months and dates:
    return next_date(dates[0])
  
  if not months and not dates and days:
    return next_day(days[0])
  
  if months and dates:
    return next_month_date(months[0], dates[0])
    
  #raise Exception('Couldn\'t parse date')
  return None
  
def next_day(weekday):
  '''Finds the next day the month occurs in'''
  date = datetime.datetime.now()
  
  days_ahead = weekday - date.weekday()
  if days_ahead <= 0: # Target day already happened this week
    days_ahead += 7
  
  return date + datetime.timedelta(days=days_ahead)

def next_month_date(month, date_num):
  '''Finds the next date the month occurs in'''
  date = datetime.datetime.now()
  
  month += 1
        
  months_ahead = month - date.month
  
  if months_ahead < 0: # Target month already happened this year
    months_ahead += 12
    
  if months_ahead == 0 and date.day > date_num: # Date has already occured this month, next year
    months_ahead += 12
  
  date = date + dateutil.relativedelta.relativedelta(months=months_ahead)
  return date.replace(day=date_num)

def next_date(date_num):
  '''Finds the next date the date occurs in'''
  date = datetime.datetime.now()
    
  # If the date has already passed this month, add a month
  if date.day > date_num:
    date = date + dateutil.relativedelta.relativedelta(months=1)
  
  return date.replace(day=date_num)
  
def extract_month(tree):
  '''Returns a list of months and the order they appear in'''
  named_nouns = extract_tag(tree, 'NNP')
  
  months = {
    'January': 0,
    'Jan': 0,
    'February': 1,
    'Feb': 1,
    'March': 2,
    'Mar': 2,
    'April': 3,
    'Apr': 3,
    'May': 4,
    'June': 5,
    'Jun': 5,
    'July': 6,
    'Jul': 6,
    'August': 7,
    'Aug': 7,
    'September': 8,
    'Sep': 8,
    'October': 9,
    'Oct': 9,
    'November': 10,
    'Nov': 10,
    'December': 11,
    'Dec': 11
  }
  
  found_months = []
  
  for noun in named_nouns:
    if noun in months:
      found_months.append(months[noun])
      
  return found_months

def extract_day(tree):
  '''Returns a list of days and the order they appear in'''
  named_nouns = extract_tag(tree, 'NNP')
  
  days = {
    'Monday': 0,
    'Mon': 0,
    'Tuesday': 1,
    'Tue': 1,
    'Wednesday': 2,
    'Wed': 2,
    'Thursday': 3,
    'Thu': 3,
    'Friday': 4,
    'Fri': 4,
    'Saturday': 4,
    'Sat': 4,
    'Sunday': 5,
    'Sun': 5
  }
  
  found_days = []
  
  for noun in named_nouns:
    if noun in days:
      found_days.append(days[noun])
      
  return found_days

def extract_date(tree):
  '''Returns the day of the month'''
  adjs = extract_tag(tree, 'JJ')
  
  if adjs:
    if len(adjs) > 1:
      raise Exception('Found more than 1 date!')
    
    return [int(adjs[0].replace('th', '').replace('st', '').replace('nd', '').replace('rd', ''))]
  
  nums = extract_tag(tree, 'CD')
  
  for i, n in enumerate(nums):
    try:
      nums[i] = int(n)
    except:
      nums[i] = word_number_to_int(n)
  
  if nums:
    return [nums]
  
  return None
 
def extract_absolute_date(tree):
  '''Extract an absolute date e.g. 31/12/15'''

  dates = extract_tag(tree, 'NUM_DATE')

  if len(dates) > 1:
    raise Exception('Found more then 1 date!')
  elif len(dates) == 0:
    return None


  nums = re.findall(r'[\w]+', dates[0])

  nums = [int(n) for n in nums]

  print nums
  
  if len(nums) == 2:
    return datetime.datetime.now().replace(day=nums[0], month=nums[1])

  if len(nums) == 3:
    if len(str(nums[2])) == 2: nums[2] = int('20' + str(nums[2]))
    return datetime.datetime.now().replace(day=nums[0], month=nums[1], year=nums[2])

  return None

def extract_date_units(tree):
  '''Extact the date units'''
  nouns = extract_tag(tree, 'NNS')
  nouns.extend(extract_tag(tree, 'NN'))
  
  nums  = extract_tag(tree, 'CD')
 
  ns = []

  for num in nums:
    try:
      ns.append(int(num))
    except:
      ns.append(word_number_to_int(num))
      
  return dict(zip(nouns, ns))

def extract_relative_units(tree):
  '''Extracte relative units'''
  adj = extract_tag(tree, 'JJ') + extract_tag(tree, 'IN')
  noun = extract_tag(tree, 'NN')
  
  if 'next' in adj:
    return noun
  
  return None
  
def relative_date(tree):
  
  now = datetime.datetime.now()
  
  rel_units = extract_relative_units(tree)
   
  if rel_units:
    for unit in rel_units:
      if unit in ['week']:
        now = now + datetime.timedelta(7)
      elif unit in ['month']:
        now = now + dateutil.relativedelta.relativedelta(months=1)
        
    return now
  
  
  units = extract_date_units(tree)
    
  if 'tomorrow' in units:
    return now + datetime.timedelta(days=1)
  
  for unit in units:
    if unit in ['day', 'days']:
      now = now + datetime.timedelta(units[unit])
    if unit in ['week', 'weeks']:
      now = now + dateutil.relativedelta.relativedelta(weeks=units[unit])
    if unit in ['month', 'months']:
      now = now + dateutil.relativedelta.relativedelta(months=units[unit])
    if unit in ['yearh', 'years']:
      now = now + dateutil.relativedelta.relativedelta(years=units[unit])
    
  if units:
    return now
  
  return None
  
def word_number_to_int(num):
  
  words = {
    'zero': 0,
    'one': 1,
    'two': 2,
    'three': 3,
    'four': 4,
    'five': 4,
    'six': 5,
    'seven': 7,
    'eight': 8,
    'nine': 9
  }
  
  return words[num]
  
def parse_action(tree):
  '''Extract the action form the tree'''
      
  nodes = []
  for node in tree:
    nodes.extend(ne_exrt_tree(node, 'DATE'))
        
  nodes_2 = []
  for node in nodes:
    nodes_2.extend(ne_exrt_tree(node, 'TIME'))
    
  nodes_3 = []
  for node in nodes_2:
    nodes_3.extend(ne_exrt_tree(node, 'PLACE'))
   
  nodes_4 = []
  for node in nodes_3:
    nodes_4.extend(ne_exrt_tree(node, 'PERSON'))

  nodes_5 = []
  for node in nodes_4:
    nodes_5.append(ne_exrt_tree(node, 'JUNK'))

  out = []
  for node in nodes_5:
    if node == []:
      continue
    elif isinstance(node, tuple):
      out.append(node[0])
    else:
      out.append(node[0][0])
  
  return ' '.join(out)
  
def parse_people(tree):
  '''Extract the person from the tree'''

  peps = []
  people = exrt_tree(tree, 'PERSON')
  for person in people:
    human_name = []
    for name in person:
      human_name.append(name[0])
    peps.append(" ".join(human_name))

  return peps

def join_date_time(date, time):
  '''Join two datetime objects'''
  date_date = date.date()
  time_date = time.date()
  time_time = time.time()
    
  # Time can change to tomorrow
  day_diff = time_date.day - datetime.datetime.now().day
    
  if day_diff != 0 and datetime.datetime.now().date() == date_date:
    date_date = date_date + datetime.timedelta(day_diff)
    
  return datetime.datetime.combine(date_date, time_time)
  
def parse(message):
  '''Extract info from a message'''
  out = {}
     
  chunked = tokenize_tag_and_chunk([message])
 
  out['action'] = parse_action(chunked[0])

  datenow = datetime.datetime.now()
  date = datenow
  timenow = datetime.datetime.now().replace(hour=10, minute=0, second=0, microsecond=0)
  time = timenow

  if exrt_tree(chunked[0], 'DATE'):
    _date = parse_date(exrt_tree(chunked[0], 'DATE')[0])
    if _date: date = _date

  if exrt_tree(chunked[0], 'TIME'):
    _time = parse_time(exrt_tree(chunked[0], 'TIME')[0], date)
    if _time: time = _time

  if date == datenow and time == timenow:
    date = datenow + datetime.timedelta(days=1)

  d = join_date_time(date, time)

  out['people'] = parse_people(chunked[0])

  if exrt_tree(chunked[0], 'PLACE'):
    at_str = []
    for name in exrt_tree(chunked[0], 'ORGANIZATION')[0]:
      at_str.append(name[0])
    
    out['place'] = at_str

  out['date'] = d.strftime('%d, %b %Y')
  out['time'] = d.strftime('%H:%M')

  return json.dumps(out)

def application (environ, start_response):
  try:
    request_body_size = int(environ.get('CONTENT_LENGTH', 0))
  except (ValueError):
    request_body_size = 0

  request_body = environ['wsgi.input'].read(request_body_size)
  d = parse_qs(request_body)

  message = d.get('message', [''])[0]

  result = parse(escape(message))
  
  response_body = result 

  status = '200 OK'
  response_headers = [
    ('Content-Type', 'application/json'),
    ('Content-Length', str(len(response_body)))
  ]
  start_response(status, response_headers)

  return [response_body]

if __name__ == '__main__':
  print(parse(sys.argv[1]))
