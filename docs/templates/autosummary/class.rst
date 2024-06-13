{{ objname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ fullname }}
   :no-members:
   :no-inherited-members:
   :no-special-members:

   {% block properties %}
   {% if attributes %}

   .. raw:: html

      <div style="background-color: #f4f4f4; padding: 0px; margin-bottom: 0px; margin-top: 2em">
         <strong>Properties:</strong>
      </div>

   {% for item in all_attributes %}
   {%- if not item.startswith('_') %}
   .. autoattribute:: {{ name }}.{{ item }}
   {%- endif -%}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block methods %}
   {% if methods %}

   .. raw:: html

      <div style="background-color: #f4f4f4; padding: 0px; margin-bottom: 0px; margin-top: 2em">
         <strong>Methods:</strong>
      </div>

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
