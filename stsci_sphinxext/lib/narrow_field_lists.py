"""
This extension exists only to tweak a docutils setting
`field_name_limit`.  `field_name_limit` is the number of characters
allowed in the field name before they are given their own row.  In the
Numpy documentation standard, "field names" are the sections headers
like "Parameters:" or "Returns:".  By setting `field_name_limit` to 1,
all of these section headers will be given their own line, those not
wasting so much horizontal space.

This extension is a little convoluted, since Sphinx doesn't appear to
provide a way to set docutils settings from the conf.py, so we need to
wait until we get a docutils builder (the builder_inited event), and
then tweak the setting directly.
"""

def setup(app):
    app.connect("builder-inited", builder_inited)

def builder_inited(app):
    app.builder.env.settings['field_name_limit'] = 1
