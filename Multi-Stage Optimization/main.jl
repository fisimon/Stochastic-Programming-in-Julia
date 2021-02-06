using JuMP
using GLPK
using DelimitedFiles
using GraphRecipes
using Plots


model = Model(GLPK.Optimizer)


ξ = readdlm("Xi.dat", '\t', Float64, '\n')
P = readdlm("P.dat", '\t', Float64, '\n')
p = readdlm("p.dat", '\t', Float64, '\n')
h = readdlm("h.dat", '\t', Float64, '\n')

N, T = size(ξ)

@variable(model, x[i=1:T+1, j=1:N^(i-1)] >=0)

@variable(model, y[i=1:T+1, j=1:N^(i-1)] >=0)


scenarios = 1:N^T

function scenario_variables(s)
    aux = []
    for t=T+1:-1:1
        push!(aux, [t, s])
        s = Int(ceil(s / 2))
    end
    return reverse(aux)
end

function scenario_rv(s)
    aux = []
    for t=T:-1:1
        push!(aux, [(s-1)%2 + 1, t])
        s = Int(ceil(s / 2))
    end
    return reverse(aux)
end




@constraint(model, y_edge, y[1,1] == 0)
#@constraint(model, x_edge[j=1:N^(T)], x[T+1, j] == 0)


for s in scenarios
    sv = scenario_variables(s)
    rv = scenario_rv(s)
    for t=1:T
        @constraint(model, x[sv[t]...] + y[sv[t]...] == y[sv[t+1]...] + ξ[rv[t]...] )
    end
end

function f(x,y)
    f = 0
    for s in scenarios
        aux = 0
        sv = scenario_variables(s)
        rv = scenario_rv(s)
        for t=1:T
            aux = aux + p[t]*x[sv[t]...] + h[t] * y[sv[t]...]
        end
        aux = aux + h[T] * y[sv[T+1]...]
        f = f + aux * prod([P[a...] for a in rv])
    end
    return f
end


@objective(model, Min, f(x,y))

optimize!(model)

termination_status(model)

objective_value(model)


res_x = []
res_y = []

edgel = Dict()
j=1
for t in 1:T+1
    for n in 1:N^(t-1)
        push!(res_x, value(x[t,n]))
        push!(res_y, value(y[t,n]))
        if t<T+1
            for i in 1:N
                edgel[(j,2*j + i -1)] = "ξ= " *string(ξ[i,t])
            end
            j= j+1
        end
    end
end


n = length(res_x)

A=zeros(n,n)

j= 2
for i=1:Int(floor(n/N))
    A[i, j:j+N-1] .= 1
    j = j+2
end

nn = "x = " .* string.(res_x) .* "\n y = " .* string.(res_y)



default(size=(2000, 2000))

graphplot(A, curves=false,
          axis_buffer=0.15,
          fontsize= 10,
          root=:right,
          nodesize = 0.02,
          nodeshape = :circle,
          method=:tree,
          edgelabel = edgel,
          names = nn)
