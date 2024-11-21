using PositionalEmbeddings
using Documenter

DocMeta.setdocmeta!(PositionalEmbeddings, :DocTestSetup, :(using PositionalEmbeddings); recursive=true)

makedocs(;
    modules=[PositionalEmbeddings],
    authors="Mateusz Kaduk <mateusz.kaduk@gmail.com> and contributors",
    sitename="PositionalEmbeddings.jl",
    format=Documenter.HTML(;
        canonical="https://mashu.github.io/PositionalEmbeddings.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/mashu/PositionalEmbeddings.jl",
    devbranch="main",
)
