
# Brat conversion schemas
# A schema is a dictionary with the following structure:
#  'name':str,   # the name of the schema
#  'desc':str,   # a textual description of why this schema exists
#  'skip_if_absent':bool,   # True whether entities not present in the schema should be skipped
#                           # False if these entities should be reproduced as-is in the converted dataset
#  'mapping': dict
# The 'mapping' dictionary has the following structure:
#    key: (entity_name, attribute_name, attribute_value)   # the entities to convert
#                                                          # all entities matching this entity name with
#                                                          # this attribute name and value will be converted.
#                                                          # "*" = any value
#                                                          # "" = no value (for attributes)
#                                                          # e.g. ("sosy", "*", "*") means all "sosy", whatever their attributes
#    value: (entity_name, attribute_name, attribute_value) # the new entity to produced
#                                                          # with the same boundaries
#                                                          # if entity_name = None -> same value as the original mention
#                                                          # if attribute_name = None -> same attribute as the original mention
#                                                          # if attribute_name = ''   -> no attribute
#                                                          # if attribute_value = None -> same attribute value
# Each schema must follow this structure
# The final BRAT_CONVERSION_SCHEMAS variable is a dictionary with
#   - key = name of the schema
#   - value = schema

sosy_and_pathologies_with_all_attributes = {
    'name': 'sosy_and_pathologies_with_all_attributes',
    'desc': '"sosy" and "pathology" are NOT grouped together into "sosypath", they are left alone, and each'
            '"assertion" attribute leads to a distinct entity. The result is '
            '"sosy" (no assertion attribute), "sosy_absent", "sosy_non_associe", "sosy_hypothetique",'
            '"pathologie" (no assertion attribute), "pathologie_absent", "pathologie_non_associe", "pathologie_hypothetique"',    
    'skip_if_absent': True,
    'mapping': {
        ('sosy', "", ""): ('sosy', '', ''),
        ('sosy', 'assertion', 'absent'): ('sosy_absent', '', ''),
        ('sosy', 'assertion', 'hypothétique'): ('sosy_hypothetique', '', ''),
        ('sosy', 'assertion', 'non-associé'): ('sosy_non_associe', '', ''),
        ('pathologie', "", ""): ('pathologie', '', ''),
        ('pathologie', 'assertion', 'absent'): ('pathologie_absent', '', ''),
        ('pathologie', 'assertion', 'hypothétique'): ('pathologie_hypothetique', '', ''),
        ('pathologie', 'assertion', 'non-associé'): ('pathologie_non_associe', '', ''),
        ('sosy', '*', '*'): ('sosy', '', ''),
        ('pathologie', '*', '*'): ('pathologie', '', '')        
    }
}
sosy_and_pathologies_with_aggregated_attributes = {
    'name': 'sosy_and_pathologies_with_aggregated_attributes',
    'desc': '"sosy" and "pathology" are NOT grouped together into "sosypath", they are left alone, and each'
            '"assertion" attribute leads to an entity sosy_nonfactual and pathology_nonfactual. The result is '
            '"sosy" (no assertion attribute), "sosy_nonfactual",'
            '"pathologie" (no assertion attribute), "pathologie_nonfactual"',    
    'skip_if_absent': True,
    'mapping': {
        ('sosy', "", ""): ('sosy', '', ''),
        ('sosy', 'assertion', '*'): ('sosy_nonfactual', '', ''),
        ('pathologie', "", ""): ('pathologie', '', ''),
        ('pathologie', 'assertion', '*'): ('pathologie_nonfactual', '', ''),
        ('sosy', '*', '*'): ('sosy', '', ''),
        ('pathologie', '*', '*'): ('pathologie', '', '')
    }
}


sosy_path_with_all_attributes = {
    'name': 'sosy_path_with_all_attributes',
    'desc': '"sosy" and "pathology" are grouped together into "sosypath", and each'
            '"assertion" attribute leads to a distinct entity. The result is '
            '"sosypath" (no assertion attribute), "sosypath_absent", "sosypath_non_associe", "sosypath_hypothetique"',
    'skip_if_absent': True,
    'mapping': {
        ('sosy', "", ""): ('sosypath', '', ''),
        ('sosy', 'assertion', 'absent'): ('sosypath_absent', '', ''),
        ('sosy', 'assertion', 'hypothétique'): ('sosypath_hypothetique', '', ''),
        ('sosy', 'assertion', 'non-associé'): ('sosypath_non_associe', '', ''),
        ('pathologie', "", ""): ('sosypath', '', ''),
        ('pathologie', 'assertion', 'absent'): ('sosypath_absent', '', ''),
        ('pathologie', 'assertion', 'hypothétique'): ('sosypath_hypothetique', '', ''),
        ('pathologie', 'assertion', 'non-associé'): ('sosypath_non_associe', '', ''),
        ('sosy', '*', '*'): ('sosypath', '', ''),
        ('pathologie', '*', '*'): ('sosypath', '', '')        
    }
}
sosy_path_with_aggregated_attributes = {
    'name': 'sosy_path_with_aggregated_attributes',
    'desc': '"sosy" and "pathology" are grouped together into "sosypath", and each'
            '"assertion" attribute leads to an entity sosypath_nonfactual. The result is '
            '"sosypath" (no assertion attribute), "sosypath_nonfactual" (all others)',
    'skip_if_absent': True,
    'mapping': {
        ('sosy', "", ""): ('sosypath', '', ''),
        ('sosy', 'assertion', '*'): ('sosypath_nonfactual', '', ''),
        ('pathologie', "", ""): ('sosypath', '', ''),
        ('pathologie', 'assertion', '*'): ('sosypath_nonfactual', '', ''),
        ('sosy', '*', '*'): ('sosypath', '', ''),
        ('pathologie', '*', '*'): ('sosypath', '', '')
    }
}

sosy_path_all = {
    'name': 'sosy_path_all',
    'desc': '"sosy" and "pathology" are grouped together into "sosypath", and each'
            '"assertion" attribute leads to an entity "sysopath" as well. The result is '
            '"sosypath" only',
    'skip_if_absent': True,
    'mapping': {
        ('sosy', "", ""): ('sosypath', '', ''),
        ('sosy', '*', '*'): ('sosypath', '', ''),
        ('pathologie', "", ""): ('sosypath', '', ''),
        ('pathologie', '*', '*'): ('sosypath', '', '')
    }
}

sosy_path_and_mesh = {
    'name': 'sosy_path_and_mesh',
    'desc': '"sosy" and "pathology" are grouped together into "sosypath", and each'
            '"assertion" attribute leads to a distinct entity. Mesh codes are added as well. \n'
            'The result is the 23 Mesh codes + '
            '"sosypath" (no assertion attribute), "sosypath_absent", "sosypath_non_associe", "sosypath_hypothetique"',
    'skip_if_absent': True,
    'mapping': {
        ('sosy', "", ""): ('sosypath', '', ''),
        ('sosy', 'assertion', 'absent'): ('sosypath_absent', '', ''),
        ('sosy', 'assertion', 'hypothétique'): ('sosypath_hypothetique', '', ''),
        ('sosy', 'assertion', 'non-associé'): ('sosypath_non_associe', '', ''),
        ('pathologie', "", ""): ('sosypath', '', ''),
        ('pathologie', 'assertion', 'absent'): ('sosypath_absent', '', ''),
        ('pathologie', 'assertion', 'hypothétique'): ('sosypath_hypothetique', '', ''),
        ('pathologie', 'assertion', 'non-associé'): ('sosypath_non_associe', '', ''),
        ('sosy', '*', '*'): ('sosypath', '', ''),
        ('pathologie', '*', '*'): ('sosypath', '', ''),
        ('ORL', '*', '*'): ('ORL', None, None),
        ('blessures', '*', '*'): ('blessures', None, None),
        ('cardiovasculaires', '*', '*'): ('cardiovasculaires', None, None),
        ('chimiques', '*', '*'): ('chimiques', None, None),
        ('digestif', '*', '*'): ('digestif', None, None),
        ('endocriniennes', '*', '*'): ('endocriniennes', None, None),
        ('etatsosy', '*', '*'): ('etatsosy', None, None),
        ('femme', '*', '*'): ('femme', None, None),
        ('genetique', '*', '*'): ('genetique', None, None),
        ('hemopathies', '*', '*'): ('hemopathies', None, None),
        ('homme', '*', '*'): ('homme', None, None),
        ('immunitaire', '*', '*'): ('immunitaire', None, None),
        ('infections', '*', '*'): ('infections', None, None),
        ('nerveux', '*', '*'): ('nerveux', None, None),
        ('nutritionnelles', '*', '*'): ('nutritionnelles', None, None),
        ('oeil', '*', '*'): ('oeil', None, None),
        ('osteomusculaires', '*', '*'): ('osteomusculaires', None, None),
        ('parasitaires', '*', '*'): ('parasitaires', None, None),
        ('peau', '*', '*'): ('peau', None, None),
        ('respiratoire', '*', '*'): ('respiratoire', None, None),
        ('stomatognathique', '*', '*'): ('stomatognathique', None, None),
        ('tumeur', '*', '*'): ('tumeur', None, None),
        ('virales', '*', '*'): ('virales', None, None)    
    }
}



BRAT_CONVERSION_SCHEMAS = {
    schema['name']:schema for schema in [sosy_path_with_all_attributes, 
                                         sosy_path_with_aggregated_attributes,
                                         sosy_path_all,
                                         sosy_path_and_mesh,
                                         sosy_and_pathologies_with_all_attributes,
                                         sosy_and_pathologies_with_aggregated_attributes]
}