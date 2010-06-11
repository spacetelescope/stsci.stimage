"""
A custom directive to produce an abstract.

It takes an author and date as options.  These are printed above the
abstract in HTML, but not printed at all in LaTeX, since the author
and date are included by another mechanism in LaTeX.

Example::

  .. abstract::
     :author: Roy G. Biv
     :date: January 1, 2000

     This paper is about the interesting properties of colors.
"""

from docutils.parsers.rst.directives.admonitions import BaseAdmonition
from docutils.parsers.rst import directives
from docutils.nodes import Admonition, Element, title

class abstract(Admonition, Element):
    pass

class AbstractDirective(BaseAdmonition):
    node_class = abstract

    option_spec = {
        'author': directives.unchanged,
        'date'  : directives.unchanged
        }

    def run(self):
        nodes = BaseAdmonition.run(self)
        nodes[0].options = self.options
        return nodes

def visit_abstract_node_html(self, node):
    if 'author' in node.options:
        self.body.append('<p><b>Author:</b> %s</p>' % node.options['author'])
    if 'date' in node.options:
        self.body.append('<p><b>Date:</b> %s</p>' % node.options['date'])

    self.body.append(self.starttag(node, 'div', CLASS=('admonition abstract')))
    node.insert(0, title('abstract', 'Abstract'))
    self.set_first_last(node)

def depart_abstract_node_html(self, node):
    self.depart_admonition(node)

def visit_abstract_node_latex(self, node):
    self.body.append("\n\\begin{abstract}\n")

def depart_abstract_node_latex(self, node):
    self.body.append("\n\\end{abstract}\n")

def visit_abstract_node_text(self, node):
    self.new_state(2)

def depart_abstract_node_text(self, node):
    self.end_state(first="Abstract: ")

def setup(app):
    app.add_node(abstract,
                 html=(visit_abstract_node_html, depart_abstract_node_html),
                 latex=(visit_abstract_node_latex, depart_abstract_node_latex),
                 text=(visit_abstract_node_text, depart_abstract_node_text))

    app.add_directive('abstract', AbstractDirective)
