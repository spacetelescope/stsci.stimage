import sample_package
import os

def show_file(dir, name) :
    list = [ os.path.dirname(__file__) ]
    if dir :
        list.append(dir)
    list.append(name)
    data_file =  os.path.join( * list )

    print "Data file in ",data_file
    f = open(data_file,"r")
    print f.read()
    f.close()
    

def run() :
    print ""
    print "Locating data files installed with the package"
    print ""
    show_file(None, "a.txt")
    print ""
    show_file("data","1.dat")
    print ""
    show_file("data","2.dat")
    print ""
    print "Look, I can sscanf:"
    import sample_package.sscanf
    print sample_package.sscanf.sscanf("001 002 003", "%d %s %d")
    print ""
    print "I am version",sample_package.__version__
    print "I am from svn:",sample_package.__svn_version__
    print "I am from:",sample_package.__full_svn_info__
