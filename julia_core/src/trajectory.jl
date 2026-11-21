# CryoMorph: Conformational Free Energy Trajectory Solver (Julia Core)
# Implements the String Method for Minimum Free Energy Paths (MFEP)
# and Overdamped Langevin Dynamics on Learned Latent Energy Landscapes U(z)

using LinearAlgebra

"""
    ConformationalLandscape

Temsil edilen latent serbest enerji peyzajı U(z) = -ln p(z).
İki kararlı durum (State A ve State B) arasındaki enerji bariyerini ve gradyanı modeller.
"""
struct ConformationalLandscape
    min_a::Vector{Float64}
    min_b::Vector{Float64}
    barrier_height::Float64
end

ConformationalLandscape() = ConformationalLandscape([-0.8, -0.5], [0.8, 0.6], 4.5)

"""
    potential_energy(landscape, z)

z noktasındaki serbest enerji değerini U(z) [k_B T biriminde] hesaplar.
"""
function potential_energy(landscape::ConformationalLandscape, z::Vector{Float64})
    da2 = sum((z .- landscape.min_a) .^ 2)
    db2 = sum((z .- landscape.min_b) .^ 2)
    
    well_a = 0.5 * da2
    well_b = 0.5 * db2
    coupling = -log(exp(-well_a) + exp(-well_b) + 1e-12)
    
    mid = 0.5 * (landscape.min_a .+ landscape.min_b)
    barrier = landscape.barrier_height * exp(-sum((z .- mid).^2) / 0.3)
    
    return coupling + barrier
end

"""
    energy_gradient(landscape, z, eps=1e-5)

U(z) potansiyelinin z noktasındaki analitik/sayısal gradyanı ∇U(z).
"""
function energy_gradient(landscape::ConformationalLandscape, z::Vector{Float64}, eps::Float64=1e-5)
    grad = zeros(length(z))
    for i in 1:length(z)
        z_plus = copy(z); z_plus[i] += eps
        z_minus = copy(z); z_minus[i] -= eps
        grad[i] = (potential_energy(landscape, z_plus) - potential_energy(landscape, z_minus)) / (2 * eps)
    end
    return grad
end

"""
    solve_minimum_energy_path(landscape, start_z, end_z; num_nodes=30, max_iters=150, dt=0.02)

String Method algoritmasıyla iki durum arasındaki En Düşük Serbest Enerji Yolunu (MFEP) bulur.
"""
function solve_minimum_energy_path(
    landscape::ConformationalLandscape,
    start_z::Vector{Float64},
    end_z::Vector{Float64};
    num_nodes::Int=30,
    max_iters::Int=150,
    dt::Float64=0.02,
    tol::Float64=1e-4
)
    string_nodes = [start_z .+ (t / (num_nodes - 1)) .* (end_z .- start_z) for t in 0:(num_nodes-1)]
    
    println("--- CryoMorph Julia String Method MFEP Solver ---")
    println("Başlangıç Düğüm Sayısı: $num_nodes | Zaman Adımı (dt): $dt")

    for iter in 1:max_iters
        old_nodes = deepcopy(string_nodes)
        
        # A. Evrim Adımı: Uç noktalar hariç her düğümü -∇U doğrultusunda kaydır
        for i in 2:(num_nodes-1)
            grad = energy_gradient(landscape, string_nodes[i])
            string_nodes[i] .-= dt .* grad
        end
        
        # B. Yeniden Parametrelendirme (Eşit Yay Uzunluğu İnterpolasyonu)
        arc_lengths = zeros(num_nodes)
        for i in 2:num_nodes
            arc_lengths[i] = arc_lengths[i-1] + norm(string_nodes[i] .- string_nodes[i-1])
        end
        total_length = arc_lengths[end]
        normalized_s = arc_lengths ./ (total_length + 1e-12)
        
        target_s = range(0.0, 1.0, length=num_nodes)
        reparam_nodes = [copy(start_z)]
        
        for k in 2:(num_nodes-1)
            s_target = target_s[k]
            idx = searchsortedlast(normalized_s, s_target)
            idx = clamp(idx, 1, num_nodes - 1)
            
            ds = normalized_s[idx+1] - normalized_s[idx]
            weight = (ds > 1e-9) ? (s_target - normalized_s[idx]) / ds : 0.0
            new_node = string_nodes[idx] .+ weight .* (string_nodes[idx+1] .- string_nodes[idx])
            push!(reparam_nodes, new_node)
        end
        push!(reparam_nodes, copy(end_z))
        string_nodes = reparam_nodes
        
        max_shift = maximum([norm(string_nodes[i] .- old_nodes[i]) for i in 1:num_nodes])
        if max_shift < tol
            println("String Method $iter iterasyonda yakınsadı (Max Shift: $(round(max_shift, digits=6))).")
            break
        end
    end

    energies = [potential_energy(landscape, node) for node in string_nodes]
    rel_energies = energies .- energies[1]
    barrier_height = maximum(rel_energies)

    println("Minimum Serbest Enerji Yolu Çözüldü. Reaksiyon Bariyeri ΔG‡ = $(round(barrier_height, digits=2)) k_B*T")
    return (path=string_nodes, free_energy_profile=rel_energies, barrier=barrier_height)
end
