#!/usr/bin/julia

# Change working directory to test dir
cd(joinpath(@__DIR__, "..", "test"))

tests = map(realpath, readlines(`find . -name "*.jl"`))
julia = joinpath(Sys.BINDIR, Base.julia_exename())

import Base.Threads: @threads

@threads for test in tests
    display(test)
    runlines = String[]
    content = String[]
    open(test, "r") do io
        last_empty = false
        while !eof(io)
            line = readline(io)
            # Skip multiple empty lines
            if isempty(line)
                if last_empty
                    continue
                else
                    last_empty = true
                end
            else
                last_empty = false
            end

            if startswith(line, r"#\s*+RUN:\s*")
                push!(runlines, line)
            elseif startswith(line, r"#\s*CHECK")
                # ignore
            else 
                push!(content, line)
            end
        end
    end
    @assert length(runlines) == 1
    runline = first(runlines)
    m = match(r"#\s*RUN:\s*(.*?)\s?\|.*", runline)
    @assert m !== nothing
    cmd = m.captures[1]
    cmd = replace(cmd, "%s"    => test)
    cmd = replace(cmd, "julia" => julia)
    cmd = replace(cmd, "2>&1" => "")

    in = IOBuffer()
    process = run(pipeline(Cmd(Base.shell_split(cmd)), stdout=in, stderr=in))
    seekstart(in)

    if !success(process)
        @error "Updating test failed" test
        write(stderr, in)
        continue
    end

    fname, io = mktemp()

    write(io, runline, "\n")
    for line in content
        write(io, line, "\n")
    end

    first_check = true
    while !eof(in) 
        line = readline(in)
        isempty(line) && continue

        if startswith(line, "module {")
            first_check = true
        elseif startswith(line, "after" ) || contains(line, r"loc\(")
            first_check = true
            continue
        end

        if first_check
            println(io)
            write(io, "# CHECK: ")
            first_check = false
        else
            write(io, "# CHECK-NEXT: ")
        end

        # These are runtime constants
        line = replace(line, r"llvm\.mlir\.constant\(\d+ : i64\)" => "llvm.mlir.constant({{[0-9]+}} : i64)")

        write(io, line, "\n")
    end
    close(io)

    mv(fname, test, force=true)
end
