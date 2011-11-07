""" This module contains the Node, Edge, and Path
  helper classes used with a Graph class that supports
  segmented graph tables with a recursion functionality
  for generating all possible obsmodes supported by
  the table.
  """
from __future__ import division
from collections import defaultdict
import os
import numpy
import pyfits
#from pysynphot import locations

_stis_spec_modes = \
['s005x29','s005x31nda','s005x31ndb','s009x29','s01x003','s01x006','s01x009','s01x02','s02x005nd',
's02x005nd','s02x006','s02x006fpa','s02x006fpb','s02x006fpc','s02x006fpd','s02x006fpe','s02x009','s02x02',
's02x02','s02x02fpa','s02x02fpb','s02x02fpc','s02x02fpd','s02x02fpe','s02x05','s02x29','s03x005nd',
's03x005nd','s03x006','s03x009','s03x02','s05x05','s10x006','s10x02','s2x2','s31x005nda',
's31x005nda','s31x005ndb','s31x005ndc','s36x005n45','s36x005p45','s36x06n45','s36x06p45','s52x005','s52x01',
's52x01','s52x02','s52x05','s52x2','s6x006','s6x02','s6x05','s6x6','e140h',
'e140h','e140hb','e140m','e140mb','e230h','e230m','g140l','g140lb','g140m',
'g140m','g140mb','g230l','g230lb','g230m','g230mb','g430l','g430m','g750l',
'g750l','g750m','prism','x140','x140m','x230','x230h','1687',
'1687','1769','1851','1884','1933','2014','2095','2176','2257',
'2257','2338','2419','2499','2579','2600','2659','2739','2800',
'2800','2818','2828','2898','2977','3055','3134','all','c1687',
'c1687','c1769','c1851','c1933','c2014','c2095','c2176','c2257','c2338',
'c2338','c2419','c2499','c2579','c2659','c2739','c2818','c2898','c2977',
'c2977','c3055','c3134','i1884','i2600','i2800','i2828','1713','1854',
'1854','1995','2135','2276','2416','2557','2697','2794','2836',
'2836','2976','3115','c1713','c1854','c1995','c2135','c2276','c2416',
'c2416','c2557','c2697','c2836','c2976','c3115','i2794','1978','2124',
'2124','2269','2415','2561','2707','c1978','c2707','ech','i2124',
'i2124','i2269','i2415','i2561','100','101','102','103','104',
'104','105','106','107','108','109','110','111','112',
'112','113','114','115','116','117','118','119','120',
'120','121','122','123','65','66','67','68','69',
'69','70','71','72','73','74','75','76','77',
'77','78','79','80','81','82','83','84','85',
'85','86','87','88','89','90','91','92','93',
'93','94','95','96','97','98','99','1763','1813',
'1813','1863','1913','1963','2013','2063','2113','2163','2213',
'2213','2263','2313','2363','2413','2463','2513','2563','2613',
'2613','2663','2713','2762','2812','2862','2912','2962','3012',
'3012','c1763','c2013','c2263','c2513','c2762','c3012','i1813','i1863',
'i1863','i1913','i1963','i2063','i2113','i2163','i2213','i2313','i2363',
'i2363','i2413','i2463','i2563','i2613','i2663','i2713','i2812','i2862',
'i2862','i2912','i2962','248','249','250','251','252','253',
'253','254','255','256','257','258','259','260','261',
'261','262','263','264','265','266','267','268','269',
'269','270','271','272','273','274','275','276','277',
'277','278','279','280','281','282','283','284','285',
'285','286','287','288','289','290','291','292','293',
'293','294','295','296','297','298','299','300','301',
'301','302','303','304','305','306','307','308','309',
'309','310','311','312','313','314','315','316','317',
'317','318','319','320','321','322','323','324','325',
'325','326','327','328','329','330','331','332','333',
'333','334','335','336','337','338','339','340','341',
'341','342','343','344','345','346','347','348','349',
'349','350','351','352','353','354','355','356','357',
'357','358','359','360','361','362','363','364','365',
'365','366','367','368','369','370','371','372','373',
'373','374','375','376','377','378','379','380','381',
'381','382','383','384','385','386','387','388','389',
'389','390','391','392','393','394','395','396','397',
'397','398','399','400','401','402','403','404','405',
'405','406','407','408','409','410','411','412','413',
'413','414','415','416','417','418','419','420','421',
'421','422','423','424','425','426','427','428','429',
'429','430','431','432','433','434','435','436','437',
'437','438','439','440','441','442','443','444','445',
'445','446','447','448','449','450','451','452','453',
'453','454','455','456','457','458','459','460','461',
'461','462','463','464','465','466','3165','3305','3423',
'3423','3680','3843','3936','4194','4451','4706','4781','4961',
'4961','5093','5216','5471','c3165','c3423','c3680','c3936','c4194',
'c4194','c4451','c4706','c4961','c5216','c5471','i3305','i3843','i4781',
'i4781','i5093','1173','1218','1222','1272','1321','1371','1387',
'1387','1400','1420','1470','1518','1540','1550','1567','1616',
'1616','1640','1665','1714','c1173','c1222','c1272','c1321','c1371',
'c1371','c1420','c1470','c1518','c1567','c1616','c1665','c1714','i1218',
'i1218','i1387','i1400','i1540','i1550','i1640','1425','c1425','1234',
'1234','1271','1307','1343','1380','1416','1453','1489','1526',
'1526','1562','1598','c1234','c1416','c1598','i1271','i1307','i1343',
'i1343','i1380','i1453','i1489','i1526','i1562','247','7751','8975',
'8975','c7751','c8975','10363','10871','5734','6094','6252','6581',
'6581','6768','7283','7795','8311','8561','8825','9286','9336',
'9336','9806','9851','c10363','c10871','c5734','c6252','c6768','c7283',
'c7283','c7795','c8311','c8825','c9336','c9851','i6094','i6581','i8561',
'i8561','i9286','i9806',
'a2d1','a2d2','a2d4','a2d8','acq']

_cos_spec_modes = \
['g130m','g140l','g160m','c1055','c1096','c1291','c1300','c1309','c1318',
'c1318','c1327','c1577','c1589','c1600','c1611','c1623','c1105','c1230',
'c1230','c1280','g185m','g225m','g230l','g285m','c1786','c1817','c1835',
'c1835','c1850','c1864','c1882','c1890','c1900','c1913','c1921','c1941',
'c1941','c1953','c1971','c1986','c2010','c2186','c2217','c2233','c2250',
'c2250','c2268','c2283','c2306','c2325','c2339','c2357','c2373','c2390',
'c2390','c2410','c2617','c2637','c2657','c2676','c2695','c2709','c2719',
'c2719','c2739','c2850','c2952','c2979','c2996','c3018','c3035','c3057',
'c3057','c3074','c3094','c2635','c2950','c3000','c3360']

rules_dict = {
  'default': {'junk': ['None', 'default', ''], 'exclude': ['aper#','aper','mjd']},
  'acs,hrc': {'junk': [], 'exclude': ['fr388n','fr459n','fr505n',
                                      'fr656n','fr914m','aper#']},
  'acs,hrc,f555w': {'junk': [], 'exclude': []},
  'acs,sbc': {'junk': [], 'exclude': []},
  'acs,wfc1': {'junk': ['wfc1'], 'exclude': ['wfc2','fr1016n','fr388n',
                                              'fr423n','fr459m','fr505n','fr656n',
                                              'fr551n','fr601n','fr647m','fr716n',
                                              'fr782n','fr853n','fr914m','fr931n',
                                              'aper#']},
  'acs,wfc2': {'junk': ['wfc2'], 'exclude': ['wfc1','fr1016n','fr388n',
                                              'fr423n','fr459m','fr505n','fr656n',
                                              'fr551n','fr601n','fr647m','fr716n',
                                              'fr782n','fr853n','fr914m','fr931n',
                                              'aper#']},
  'wfc3,uvis1': {'junk': ['uvis1'], 'exclude': ['uvis2','noota','ota','dn',
                                                'qyc','cal','aper#']},
  'wfc3,uvis2': {'junk': ['uvis2'], 'exclude': ['uvis1','noota','ota','dn',
                                                'qyc','cal','aper#']},
  'wfc3,ir': {'junk': [], 'exclude': ['aper#']},
  'nicmos,1': {'junk': [], 'exclude': []},
  'nicmos,2': {'junk': [], 'exclude': []},
  'nicmos,3': {'junk': [], 'exclude': []},
  'cos,*': {'junk': [], 'exclude': _cos_spec_modes},
  'wfpc2,*': {'junk': [], 'exclude': ['lrf','cont','dn']},
  'stis,*': {'junk': [], 'exclude': _stis_spec_modes}
}

class ObsmodeError(BaseException):
  pass

class Node(object):
  """A Node correspondes to a graph table 'innode'.
    Nodes are attached to each other by Edges.
    """
  def __init__(self, name):
    """name = innode""" 
    self.name = name
    self.edges = dict()
    self.edgeset = set()
  
  def __iter__(self):
    return self.edges.values().__iter__()
  
  def __str__(self):
    return str(self.name)
  
  def add_edge(self, edge):
    #The dict is used to select edge based on kwd
    self.edges[edge.kwd] = edge
    #The set is used for fast matching in .select_edge()
    self.edgeset.add(edge.kwd)
  
  def select_edge(self, kwdset, partial=False, count=0):
    # based on algorithm by Alex Martelli
    # kwdset = set of the comma-separated keywords
    # in an obsmode. eg
    # "acs,hrc,f555w" -> set(["acs","hrc","f555w"])
    match = self.edgeset & kwdset
    if len(match) > 1:
      print "Found match of ",match
      raise ValueError("Ambiguous...Too many edges match. Check for problems with graph table.")
    elif len(match) == 1:
      ans = self.edges[match.pop()]
    else:
      # pick up the default if there is one
      if 'default' in self.edges:
        #try:
        if not partial or (partial and count < len(kwdset)):
          #consider 'default' -> None
          ans = self.edges['default']
        else:
          # define Edge object to serve as sentinal to mark the end of this path
          ans = Edge('default',[None,None,None,None],None)
      #except KeyError:
      else:
        # An example of this case would be kwdset=['acs'] yet 
        # the only edges for continuing are ['wfc2','sbc','wfc1','hrc']
        #raise KeyError("No match, bla bla")
        print "No match... Multiple edges but no default."
        # define Edge object to serve as sentinal to mark the end of this path
        ans = Edge('default',[None,None,None,None],None)
    return ans

class Edge(object):
  """An Edge connects a pair of nodes. An Edge
    contains a keyword, compname, and thcompname.
    """
  def __init__(self, kwd, contents, outnode):
    self.kwd = kwd
    self.optical, self.thermal,self.filename,self.pvar=contents
    self.destination = outnode

class Path(object):
  """A Path is produced by traversing a Graph.
    It contains a list of edges and has some convenience
    methods.
    """
  
  def __init__(self, obsmode, params=None):
    self.name = obsmode
    self._edgelist = []
    self._params = params
    self.offset = None
  
  def append(self, edge):
    self._edgelist.append(edge)
    parvalues = []
    parname = edge.filename
    if edge.pvar in self._params and edge.pvar not in [None,'',' ']:
      # Get throughput file for component
      f = pyfits.open(edge.filename)
      cnames = f[1].data.names[1:]
      for pname in cnames:
        if edge.pvar in pname:
          parvalues.append(float(pname.split('#')[-1]))
      f.close()
      self._params[edge.pvar] = parvalues
  
  def __iter__(self):
    return self._edgelist.__iter__()
  
  def compnames(self):
    """Return list of optical compnames"""
    ans = [e.optical for e in self._edgelist if e.optical is not None]
    return ans

class Graph(object):
  """A Graph is the data structure that corresponds to
    the graph table.
    """
  def __init__(self, nodes=None, name=None):
    # fundamental contents
    self.nodes = dict()
    self.name = name
    # metadata usual for use with real graphtabs
    self.tmcfiles = list()
    self.tmtfiles = list()
    self.area = None
    # constructed data usual for use with real graphtabs
    self.lookup = dict()
    
    if nodes is not None:
      for k in nodes:
        self.add_node(k)
      self.top = nodes[0]
    
    self.obsmode = None
    self.obsmodes = []
  
  def __len__(self):
    return len(self.nodes)
  
  def add_node(self, node):
    self.nodes[node.name] = node
  
  def __getitem__(self, nodename):
    return self.nodes[nodename]
  
  def __contains__(self, nodename):
    return (nodename in self.nodes)
  
  def get(self, nodename, default=None):
    return self.nodes.get(nodename, default)
  
  def traverse(self, obsmode, verbose=True,partial=False):
    """ This is the usual way to traverse a graph
    table. obsmode_str is a comma-separated obsmode
    string, such as 'acs, hrc, f555w'.
    
    If 'partial' == True, logic will be followed to identify the
    point at which all components of the partial obsmode provided as input
    are found to match at least 1 keyword. 
    
    """
    
    #counter for sanity check
    count=0
    # counter for number of obsmode elements found in graphtab
    kwdcount = 0
    
    #Turn obsmode into a keyword set
    kwdset = set(obsmode.split(','))
    
    #Handle parameterized kwds here
    plist = [k for k in kwdset if '#' in k]
    params = dict()
    for p in plist:
      kwdset.remove(p)
      k,v = p.split('#')
      k += '#'  # needed to match syntax used in graphtab
      params[k.upper()] = v #floated?, always use uppercase for matching
      kwdset.add(k)
    
    #Path to accumulate answers
    ans = Path(obsmode, params=params)
    
    #Start at the top of the tree
    innode = self[self.top.name]
    if verbose:
      print "Starting: ", innode.name
    
    edge_list=[]
    
    while ( (innode is not None) and
           (count <= len(self)) ):
      # Choose edge based on obsmode
      try:
        edge = innode.select_edge(kwdset,partial=partial,count=kwdcount)
      except KeyError:
        # came to the end of the obsmode in the middle of the graph
        break
      ans.append(edge)
      
      if verbose:
        print "->".join([str(innode),
                         str(edge.kwd),
                         str(edge.destination)])
      
      if 'acs' in obsmode or 'wfc3' in obsmode:
        # Count up the number of obsmode elements accounted for in the Path
        if edge.kwd in kwdset and edge.kwd not in set(edge_list) and edge.kwd != 'default': 
          kwdcount += 1
          # Keep track of the offset to this node 
          if edge.destination is not None:
            ans.offset = edge.destination.name
      else:
        # Count up the number of obsmode elements accounted for in the Path
        if edge.kwd in kwdset and edge.kwd != 'default': 
          if edge.kwd not in set(edge_list):
            kwdcount += 1
          # Keep track of the offset to this node 
          if edge.destination is not None:
            ans.offset = edge.destination.name
      
      edge_list.append(edge.kwd)
      
      # Set up for passing through this node to the next
      innode = edge.destination
      count+=1
    
    return ans
  
  def get_obsmodes(self,obsmode,prefix=False):
    """UI to the recursive process of obtaining
    all obsmodes in the graph. First calls the
    recurse method, then calls a post-processor
    to remove Nones and duplicates.
    
    If no value is provided for `offset`, then
    it will start at the offset used to 
    read in the graph. If a string is provided
    as the value for `offset`, then .traverse().offset
    will be determined and used. 
    
    The returned obsmode strings will contain the 
    full obsmode if a string is provided for `offset`
    and `prefix`=True.
    
    """
    self.obsmode = obsmode
    
    if self.obsmodes != []:
      self.obsmodes = []
    
    if obsmode in rules_dict.keys():
      self.obsrules_key = obsmode
    elif obsmode + ',*' in rules_dict.keys():
      self.obsrules_key = obsmode + ',*'
    else:
      raise ObsmodeError('Unsupported obsmode string: ' + unicode(obsmode))
    
    self.obsrules_junk = rules_dict['default']['junk'] + \
                          rules_dict[self.obsrules_key]['junk']
    self.obsrules_excl = rules_dict['default']['exclude'] + \
                          rules_dict[self.obsrules_key]['exclude']
                          
    # make sure the given obsmode string doesn't contain anything excluded
    for o in obsmode.split(','):
      if o in self.obsrules_excl:
        s = 'Entered obsmode includes excluded parameters: ' + unicode(obsmode)
        raise ObsmodeError(s)
    
    instrmode=None
    
    if isinstance(obsmode, str):
      if prefix: 
        instrmode=obsmode
      offset = self.traverse(obsmode,verbose=False,partial=False).offset
      
    print 'Start at offset: ',offset
      
    startnode = self[offset]
    # build up raw list of all possible obsmodes
    self.obsmodes = self.recurse(startnode)
    
    # remove junk from obsmode strings
    self.process(prefix=instrmode)
    
    return self.obsmodes
#    return real_result
  
  def process(self,prefix=None):
    """
    Remove 'default' strings and put on the prefix, if supplied.
    """
    
    # convert to a numpy array so we can use fast vectorized functions
    self.obsmodes = numpy.array(self.obsmodes, dtype=numpy.str)
    
    # remove junk
#    for junk in self.obsrules_junk:
#      if junk == '':
#        j = ',,'
#        r = ','
#        self.obsmodes = numpy.char.replace(self.obsmodes, j, r)
#      else:
#        j1 = ',' + junk
#        j2 = junk + ','
#        j3 = junk
#        r = ''
#
#        self.obsmodes = numpy.char.replace(self.obsmodes, j1, r)
#        self.obsmodes = numpy.char.replace(self.obsmodes, j2, r)
#        self.obsmodes = numpy.char.replace(self.obsmodes, j3, r)
#        
#    self.obsmodes = numpy.unique(self.obsmodes)

    # SPECIAL FOR WFC3,UVIS
    # because of graph table shenanigans it easiest to generate a list of obsmodes
    # that don't include the cal keyword and then duplicate that list and append
    # "cal" to the duplicates
    if 'wfc3,uvis' in self.obsmode.lower():
      calmodes = self.obsmodes.copy()
      calmodes = numpy.char.add(calmodes,',cal')
      self.obsmodes = numpy.array(self.obsmodes.tolist() + calmodes.tolist(),dtype=numpy.str)
    
    # add prefix (or not) and convert back to list
    if prefix is not None:
      self.obsmodes = numpy.char.add(prefix,self.obsmodes).tolist()
      
      # remove trailing comma if necessary
      if self.obsmodes[0][:-1] == prefix:
        self.obsmodes[0] = prefix
    
    else:
      self.obsmodes = numpy.char.replace(self.obsmodes,',','',1).tolist()
  
  def read_rules(self, fname):
    """
      """
    # read the data file
    f = open( fname, 'r')
    
    datastr = f.read()
    f.close()
    
    # convert DOS file to Unix - otherwise the eval will fail
    datastr = datastr.replace('\r','')
    
    # try to eval the data
    try :
      datadict = eval(datastr)
    except Exception, e:
      print 'EXCEPTION:',e
      print 'cannot eval data in file ',fname
      raise
    
    return datadict
  
  def recurse(self, node, so_far=None):
    """This is the method that can be used to
    generate all possible obsmodes from a table
    (starting either from the root or from a
    specified node).

    This version returns a list of strings that
    must be processed to remove junk.
    """
    result = list()
    r=list()

    #This is syntactic sugar
    if so_far is None:
      so_far = ''            

    #This terminates the recursion
    if node is None:
#       print "Terminating at ",so_far
      result.append(so_far)

    else:
#       print "Entering node ",node.name
      #Recurse through all the edges of this node
      for edge in node:
        if edge.kwd.lower() in self.obsrules_excl:
          continue
            
        # wfpc2 obsmodes may have at most 2 filters, including polarized filters
        if 'wfpc2' in self.obsmode.lower() and \
           (edge.kwd.lower()[0] == 'f' or \
            edge.kwd.lower()[:3] == 'pol'):
          num_filt = 0
          for mode in so_far.split(','):
            if mode not in ['','default'] and \
               (mode.lower()[0] == 'f' or mode.lower()[:3] == 'pol'):
              num_filt += 1
          if num_filt >= 2:
            continue
      
        #print "Processing node ",node.name, ", edge ",edge.kwd
        if edge.kwd not in self.obsrules_junk:
          further = so_far + "," + edge.kwd
        else:
          further = so_far
        #print "Further is ",further
        #result.append(further)
        
        ans = self.recurse(edge.destination,further)
        #print "ans is ",ans

        #Collect the answers for each edge
        result.extend(ans)

    return result
