import Scripting
import os

def set_options(opt):
    opt.tool_options('compiler_cc')
    opt.tool_options('compiler_cxx')

def configure(conf):
    conf.check_tool('compiler_cc')
    conf.check_tool('compiler_cxx')

    conf.env['BUILD'] = conf.blddir

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
    Scripting.commands += ['configure', 'build', 'do_tests']

def do_tests(ctx):
    ctx.recurse("test_c")

def valgrind(ctx):
    ctx.recurse("test_c")
