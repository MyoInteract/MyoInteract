import xml.etree.ElementTree as ET

def create(root, name):
  # Creates an element if it doesn't exist
  element = root.find(name)
  if element is None:
    root.append(ET.Element(name))

def copy_or_append(name, src, dst):
  element = src.find(name)

  if dst.find(name) is None:
    dst.append(element)
  else:
    dst.find(name).append(element)

def copy_or_merge(name, classname, src, dst):
  if classname is not None:
    copy_or_merge_classname(name, classname, src, dst)
  else:
    element = src.find(name)
    if dst.find(name) is None:
      dst.append(element)
    else:
      for k, v in {*dst.find(name).items(), *element.items()}:
        dst.find(name).set(k, v)

def copy_or_merge_classname(name, classname, src, dst):
  element = src.find(f"{name}[@class='{classname}']")
  if dst.find(f"{name}[@class='{classname}']") is None:
    dst.append(element)
  else:
    for k, v in {*dst.find(f"{name}[@class='{classname}']").items(), *element.items()}:
      dst.find(f"{name}[@class='{classname}']").set(k, v)

def copy_children(name, src, dst, exclude=None):

  # Check if there is something to copy
  elements = src.find(name)

  if elements is not None:

    # Create an element if necessary
    create(dst, name)

    # Copy each element except ones that are excluded
    for element in elements:
      if exclude is not None and \
          element.tag == exclude["tag"] and element.attrib[exclude["attrib"]] == exclude["name"]:
        continue
      else:
        dst.find(name).append(element)

def merge_children(name, src, dst, exclude=None):

  # Check if there is something to copy
  src_elements = src.find(name)

  if src_elements is not None:

    # Create an element if necessary
    create(dst, name)

    dst_elements = dst.find(name)

    # Copy each element except ones that are excluded
    for element in src_elements:
      if exclude is not None and \
          element.tag == exclude["tag"] and element.attrib[exclude["attrib"]] == exclude["name"]:
        continue
      else:
        copy_or_merge(element.tag, element.get("class", None), src_elements, dst_elements)