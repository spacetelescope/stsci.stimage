import os

def options(ctx):
    ctx.load('compiler_c')

def configure(ctx):
    ctx.load('compiler_c')

def build(ctx):
    # Install header files
    start_dir = ctx.path.find_dir('include')
    ctx.install_files(
        "${PREFIX}/include",
        start_dir.ant_glob('**/*'),
        cwd=start_dir,
        relative_trick=True)

    ctx.recurse("src")
    ctx.recurse("test_c")

def test(ctx):
    from waflib import Options
    Options.commands += ['configure', 'build', 'do_tests']

def do_tests(ctx):
    ctx.recurse("test_c")

def valgrind(ctx):
    ctx.recurse("test_c")
