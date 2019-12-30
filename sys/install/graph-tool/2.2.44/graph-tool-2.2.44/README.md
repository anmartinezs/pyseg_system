graph-tool -- an efficient python module for analysis of graphs
================================================================

graph-tool is an efficient python module for manipulation and
statistical analysis of graphs. It contains several general graph
measurements, data structures and algorithms, such as vertex and edge
properties, online graph filtering, nearest neighbour statistics,
clustering, interactive graph layout, random graph generation, detection
of community structure, and more.

Contrary to most other python modules with similar functionality, the
core data structures and algorithms are implemented in C++, making
extensive use of template metaprogramming, based heavily on the Boost
Graph Library. This confers it a level of performance that is
comparable (both in memory usage and computation time) to that of a
pure C/C++ library.

For more information and documentation, please take a look at the
website http://graph-tool.skewed.de.

graph-tool is free software, you can redistribute it and/or modify it
under the terms of the GNU General Public License, version 3 or
above. See COPYING for details.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

Availability
------------

The current stable version of graph-tool is always available from the
project's website: http://graph-tool.skewed.de

Installation
------------

graph-tool follows the standard GNU installation procedure.  Please
consult the INSTALL file in this distribution for more detailed
instructions.

Note that recent versions of GCC (4.8 or above) or clang, with good
c++11 support, are required for compilation. Due to the heavy use of
template metaprogramming techniques, **relatively large amounts of RAM
are required during compilation**. You have been warned!  For this
reason, pre-compiled packages are available in the website.

More information about graph-tool
---------------------------------

The project homepage is http://graph-tool.skewed.de. It contains
documentation, info on mailing lists, as well as a bug-tracking
function. You should be reading it, instead of this. :-)

Reporting Bugs
--------------

A list of known bugs can be found in the website:

http://graph-tool.skewed.de/issues

If you found a bug in the program which is not included in this list,
please submit a ticket through the provided interface.

--
Tiago de Paula Peixoto <tiago@skewed.de>