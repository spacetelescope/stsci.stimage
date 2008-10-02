import distutils.extension
pkg = "stscidocs"


files = [ ( pkg, [ "index.html" ] ) ]

f = open("package_list","r")

for x in  f :
    x = x.strip()
    if x.startswith("#") :
        continue
    files.append( ( pkg+"/docs/"+x, [ "docs/"+x+"/*" ] ) )

for x in  [ 'MultiDrizzle_Ch1', 'SynphotManual', 'hst_synphot', 'pydatatut', 'pyfits_users_manual', 'pyraf_guide', 'pyraf_tutorial' ]:
    files.append( (pkg+"/docs/", [ "docs/"+x+".pdf" ] ) )

setupargs = {

    'version' :         '2.7',
    'author_email' :    'help@stsci.edu',
    'scripts' :         [ 'lib/stscidocs' ],
    'license' :         'BSD',
    'data_files' :      files
}

