dnl config.m4 for extension bs_matrix

dnl Comments in this file start with the string 'dnl'.
dnl Remove where necessary.

dnl If your extension references something external, use with:
dnl PHP_ARG_WITH(bs_matrix, for bs_matrix support, [  --with-bs_matrix             Include bs_matrix support])

dnl Otherwise use enable:
PHP_ARG_ENABLE(bs_matrix, whether to enable bs_matrix support, [  --enable-bs_matrix          Enable bs_matrix support], no)

if test "$PHP_BS_MATRIX" != "no"; then
  dnl Write more examples of tests here...

  dnl # get library FOO build options from pkg-config output
  dnl AC_PATH_PROG(PKG_CONFIG, pkg-config, no)
  dnl AC_MSG_CHECKING(for libfoo)
  dnl if test -x "$PKG_CONFIG" && $PKG_CONFIG --exists foo; then
  dnl   if $PKG_CONFIG foo --atleast-version 1.2.3; then
  dnl     LIBFOO_CFLAGS=\`$PKG_CONFIG foo --cflags\`
  dnl     LIBFOO_LIBDIR=\`$PKG_CONFIG foo --libs\`
  dnl     LIBFOO_VERSON=\`$PKG_CONFIG foo --modversion\`
  dnl     AC_MSG_RESULT(from pkgconfig: version $LIBFOO_VERSON)
  dnl   else
  dnl     AC_MSG_ERROR(system libfoo is too old: version 1.2.3 required)
  dnl   fi
  dnl else
  dnl   AC_MSG_ERROR(pkg-config not found)
  dnl fi
  dnl PHP_EVAL_LIBLINE($LIBFOO_LIBDIR, BS_MATRIX_SHARED_LIBADD)
  dnl PHP_EVAL_INCLINE($LIBFOO_CFLAGS)

  dnl # --with-bs_matrix -> check with-path
  dnl SEARCH_PATH="/usr/local /usr"     # you might want to change this
  dnl SEARCH_FOR="/include/bs_matrix.h"  # you most likely want to change this
  dnl if test -r $PHP_BS_MATRIX/$SEARCH_FOR; then # path given as parameter
  dnl   BS_MATRIX_DIR=$PHP_BS_MATRIX
  dnl else # search default path list
  dnl   AC_MSG_CHECKING([for bs_matrix files in default path])
  dnl   for i in $SEARCH_PATH ; do
  dnl     if test -r $i/$SEARCH_FOR; then
  dnl       BS_MATRIX_DIR=$i
  dnl       AC_MSG_RESULT(found in $i)
  dnl     fi
  dnl   done
  dnl fi
  dnl
  dnl if test -z "$BS_MATRIX_DIR"; then
  dnl   AC_MSG_RESULT([not found])
  dnl   AC_MSG_ERROR([Please reinstall the bs_matrix distribution])
  dnl fi

  dnl # --with-bs_matrix -> add include path
  dnl PHP_ADD_INCLUDE($BS_MATRIX_DIR/include)

  dnl # --with-bs_matrix -> check for lib and symbol presence
  dnl LIBNAME=BS_MATRIX # you may want to change this
  dnl LIBSYMBOL=BS_MATRIX # you most likely want to change this

  dnl PHP_CHECK_LIBRARY($LIBNAME,$LIBSYMBOL,
  dnl [
  dnl   PHP_ADD_LIBRARY_WITH_PATH($LIBNAME, $BS_MATRIX_DIR/$PHP_LIBDIR, BS_MATRIX_SHARED_LIBADD)
  dnl   AC_DEFINE(HAVE_BS_MATRIXLIB,1,[ ])
  dnl ],[
  dnl   AC_MSG_ERROR([wrong bs_matrix lib version or lib not found])
  dnl ],[
  dnl   -L$BS_MATRIX_DIR/$PHP_LIBDIR -lm
  dnl ])
  dnl
  dnl PHP_SUBST(BS_MATRIX_SHARED_LIBADD)

  dnl # In case of no dependencies
  AC_DEFINE(HAVE_BS_MATRIX, 1, [ Have bs_matrix support ])

  PHP_NEW_EXTENSION(bs_matrix, bs_matrix.c, $ext_shared)
fi
