/* config.h.  Generated from config.h.in by configure.  */
/* config.h.in.  Generated from configure.ac by autoheader.  */

/* program author(s) */
#define AUTHOR "Tiago de Paula Peixoto <tiago@skewed.de>"

/* copyright info */
#define COPYRIGHT "Copyright (C) 2006-2014 Tiago de Paula Peixoto\nThis is free software; see the source for copying conditions.  There is NO\nwarranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE."

/* c++ preprocessor compilation options */
#define CPPFLAGS "-I/fs/pool/pool-bmsan-apps/antonio/sys/soft/cgal/4.7/build/include -I/fs/pool/pool-bmsan-apps/antonio/sys/soft/cgal/4.7/infrstr/include -I/fs/pool/pool-bmsan-apps/antonio/app/soft/anaconda2/4.3.1/include/python2.7 -I/usr/include -I/fs/pool/pool-bmsan-apps/antonio/app/soft/anaconda2/4.3.1/lib/python2.7/site-packages/numpy/core/include"

/* c++ compilation options */
#define CXXFLAGS "-Wall -Wextra -ftemplate-backtrace-limit=0  -DNDEBUG -std=gnu++11 -ftemplate-depth-250 -Wno-deprecated -Wno-unknown-pragmas -O3 -fvisibility=default -fvisibility-inlines-hidden -Wno-unknown-pragmas"

/* compile debug info */
/* #undef DEBUG */

/* GCC version value */
#define GCC_VERSION (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__)

/* git HEAD commit hash */
#define GIT_COMMIT "178add3a"

/* git HEAD commit date */
#define GIT_COMMIT_DATE "Thu Jul 2 01:44:54 2015 +0200"

/* define if the Boost library is available */
#define HAVE_BOOST /**/

/* define if the Boost::Graph library is available */
#define HAVE_BOOST_GRAPH /**/

/* define if the Boost::Iostreams library is available */
#define HAVE_BOOST_IOSTREAMS /**/

/* define if the Boost::Python library is available */
#define HAVE_BOOST_PYTHON /**/

/* define if the Boost::Regex library is available */
#define HAVE_BOOST_REGEX /**/

/* Cairomm is available */
/* #undef HAVE_CAIROMM */

/* Define to 1 if you have the <dlfcn.h> header file. */
#define HAVE_DLFCN_H 1

/* Define to 1 if you have the <inttypes.h> header file. */
#define HAVE_INTTYPES_H 1

/* Define to 1 if you have the `bz2' library (-lbz2). */
#define HAVE_LIBBZ2 1

/* Define to 1 if you have the `CGAL' library (-lCGAL). */
#define HAVE_LIBCGAL 1

/* Define to 1 if you have the `expat' library (-lexpat). */
#define HAVE_LIBEXPAT 1

/* Define to 1 if you have the `m' library (-lm). */
#define HAVE_LIBM 1

/* Define to 1 if you have the <memory.h> header file. */
#define HAVE_MEMORY_H 1

/* If available, contains the Python version number currently in use. */
#define HAVE_PYTHON "2.7"

/* using scipy's weave */
/* #undef HAVE_SCIPY */

/* Using google's sparsehash */
/* #undef HAVE_SPARSEHASH */

/* Define to 1 if you have the <stdint.h> header file. */
#define HAVE_STDINT_H 1

/* Define to 1 if you have the <stdlib.h> header file. */
#define HAVE_STDLIB_H 1

/* Define to 1 if you have the <strings.h> header file. */
#define HAVE_STRINGS_H 1

/* Define to 1 if you have the <string.h> header file. */
#define HAVE_STRING_H 1

/* Define to 1 if you have the <sys/stat.h> header file. */
#define HAVE_SYS_STAT_H 1

/* Define to 1 if you have the <sys/types.h> header file. */
#define HAVE_SYS_TYPES_H 1

/* Define to 1 if you have the <unistd.h> header file. */
#define HAVE_UNISTD_H 1

/* python prefix */
#define INSTALL_PREFIX "/fs/pool/pool-bmsan-apps/antonio/sys/soft/graph-tool/2.2.44/build"

/* linker options */
#define LDFLAGS "-L/fs/pool/pool-bmsan-apps/antonio/sys/soft/cgal/4.7/build/lib -L/fs/pool/pool-bmsan-apps/antonio/sys/soft/cgal/4.7/infrstr/lib -L/fs/pool/pool-bmsan-apps/antonio/app/soft/anaconda2/4.3.1/lib -lpython2.7"

/* Define to the sub-directory where libtool stores uninstalled libraries. */
#define LT_OBJDIR ".libs/"

/* disable function inlining */
/* #undef NO_INLINE */

/* Name of package */
#define PACKAGE "graph-tool"

/* Define to the address where bug reports for this package should be sent. */
#define PACKAGE_BUGREPORT "http://graph-tool.skewed.de/tickets"

/* package data dir */
#define PACKAGE_DATA_DIR "/fs/pool/pool-bmsan-apps/antonio/sys/soft/graph-tool/2.2.44/build/share/graph-tool"

/* package doc dir */
#define PACKAGE_DOC_DIR "${datarootdir}/doc/${PACKAGE_TARNAME}"

/* Define to the full name of this package. */
#define PACKAGE_NAME "graph-tool"

/* package source dir */
#define PACKAGE_SOURCE_DIR "/fs/pool/pool-bmsan-apps/antonio/sys/soft/graph-tool/2.2.44/graph-tool-2.2.44"

/* Define to the full name and version of this package. */
#define PACKAGE_STRING "graph-tool 2.2.44"

/* Define to the one symbol short name of this package. */
#define PACKAGE_TARNAME "graph-tool"

/* Define to the home page for this package. */
#define PACKAGE_URL "http://graph-tool.skewed.de"

/* Define to the version of this package. */
#define PACKAGE_VERSION "2.2.44"

/* The directory name for the site-packages subdirectory of the standard
   Python install tree. */
#define PYTHON_DIR "/fs/pool/pool-bmsan-apps/antonio/app/soft/anaconda2/4.3.1/lib/python2.7/site-packages"

/* Sparsehash include macro */
/* #undef SPARSEHASH_INCLUDE */

/* Define to 1 if you have the ANSI C header files. */
#define STDC_HEADERS 1

/* Enable extensions on AIX 3, Interix.  */
#ifndef _ALL_SOURCE
# define _ALL_SOURCE 1
#endif
/* Enable GNU extensions on systems that have them.  */
#ifndef _GNU_SOURCE
# define _GNU_SOURCE 1
#endif
/* Enable threading extensions on Solaris.  */
#ifndef _POSIX_PTHREAD_SEMANTICS
# define _POSIX_PTHREAD_SEMANTICS 1
#endif
/* Enable extensions on HP NonStop.  */
#ifndef _TANDEM_SOURCE
# define _TANDEM_SOURCE 1
#endif
/* Enable general extensions on Solaris.  */
#ifndef __EXTENSIONS__
# define __EXTENSIONS__ 1
#endif


/* using openmp */
/* #undef USING_OPENMP */

/* Version number of package */
#define VERSION "2.2.44"

/* Define to 1 if on MINIX. */
/* #undef _MINIX */

/* Define to 2 if the system does not provide POSIX.1 features except with
   this defined. */
/* #undef _POSIX_1_SOURCE */

/* Define to 1 if you need to in order for `stat' and other things to work. */
/* #undef _POSIX_SOURCE */
