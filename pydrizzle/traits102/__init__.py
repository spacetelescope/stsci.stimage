#-------------------------------------------------------------------------------
#
#  Define a 'traits' package that allows other classes to easily define
#  'type-checked' and/or 'delegated' traits for their instances.
#
#  Note: A 'trait' is a synonym for 'property', but is used instead of the
#  word 'property' to differentiate it from the Python language 'property'
#  feature.
#
#  Written by: David C. Morrill
#
#  Date: 06/21/2002
#
#  (c) Copyright 2002 by Enthought, Inc.
#
#-------------------------------------------------------------------------------

from trait_base      import Undefined, Self, trait_editors
from trait_errors    import TraitError, DelegationError
from traits          import HasTraits, HasObjectTraits, HasDynamicTraits, \
                            HasDynamicObjectTraits, Trait, Disallow, ReadOnly, \
                            DefaultPythonTrait, TraitProxy
from trait_handlers  import TraitHandler, TraitRange, TraitType, TraitString,  \
                            TraitInstance, TraitThisClass, TraitClass, \
                            TraitFunction, TraitEnum, TraitMap
from trait_handlers  import TraitList, TraitPrefixList, TraitPrefixMap,        \
                            TraitComplex, AnyValue
from trait_delegates import TraitGetterSetter, TraitDelegate, \
                            TraitDelegateSynched, TraitEvent, TraitProperty
from trait_sheet     import TraitSheetHandler, TraitEditor, TraitGroup, \
                            TraitGroupItem, TraitGroupList, merge_trait_groups
