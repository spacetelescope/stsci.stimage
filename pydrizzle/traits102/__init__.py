#--------------------------------------------------------------------------------
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
#--------------------------------------------------------------------------------

from traits import HasTraits, HasDynamicTraits, Trait, TraitDelegate
from traits import TraitDelegateSynched, TraitHandler, TraitRange, TraitType
from traits import TraitInstance, TraitFunction, TraitEnum, TraitMap
from traits import TraitPrefixList, TraitPrefixMap, TraitComplex, Undefined
from traits import AnyValue, Disallow, ReadOnly, DefaultPythonTrait, TraitError
from traits import DelegationError, trait_editors
