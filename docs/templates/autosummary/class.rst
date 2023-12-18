{{ objname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ fullname }}
   :no-members:
   :no-inherited-members:
   :no-special-members:

   {% block properties %}
   {% if attributes %}

   **Properties**

   .. autosummary::
   {% for item in all_attributes %}
      {%- if not item.startswith('_') %}
      ~{{ name }}.{{ item }}
      {%- endif -%}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block methods %}
   {% if methods %}

   **Methods**

   .. autosummary::
      :toctree:
      :nosignatures:
   {% for item in all_methods %}
      {%- if not item.startswith('_') %}
      ~{{ name }}.{{ item }}
      {%- endif -%}
   {%- endfor %}
   {% endif %}
   {% endblock %}
