using Plots, Distributions
cd("H:\\OneDrive\\Facu\\Labo 5\\labo_5\\output") #no encontré forma mejor de hacer esto
x = range(-5, 5, 100)

abs = pdf.(Normal(0, 1), x)
abs_izq = pdf.(Normal(-1, 1.5), x)
abs_der = pdf.(Normal(1, 1.5), x)

plot(x, abs, label="Sin campo", lw=2,
 formatter=Returns(""), legendfontsize=10,
 xguidefontsize=15, yguidefontsize=15)
plot!(x, abs_izq, label="Cir. izq.", lw=2)
plot!(x, abs_der, label="Cir. der.", lw=2)
plot!(x, abs_izq-abs_der, label="Señal DAVS", lw=2)
xlabel!("Frecuencia (a.u.)")
ylabel!("Absorción (a.u.)")
savefig("absorcion.svg")