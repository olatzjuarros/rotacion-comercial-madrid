// Riesgo de rotación comercial · Madrid — JS puro, sin framework.
// Todas las llamadas son a la propia API (mismo origen), salvo las teselas
// del mapa y el script de Leaflet, que vienen de CDN.

const $ = (sel) => document.querySelector(sel);
const pct = (x) => (x * 100).toLocaleString("es-ES", { maximumFractionDigits: 2 }) + " %";
const debounce = (fn, ms) => {
  let t;
  return (...args) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...args), ms);
  };
};

// ---------------------------------------------------------------------
// Cabecera: GET /health
// ---------------------------------------------------------------------
async function cargarCabecera() {
  try {
    const r = await fetch("/health");
    if (!r.ok) return;
    const d = await r.json();
    $("#h-modelo").textContent = d.modelo;
    $("#h-auc").textContent = d.metricas_test.auc.toFixed(3);
    $("#h-base").textContent = pct(d.tasa_base);

    // Cabecera explicativa: todo numero sale de /health (== ficha_modelo.json
    // en el servidor), nunca escrito a fuego aqui.
    const cohortesTrain = (d.cohortes_train || []).join("-");
    const cohorteTest = d.cohorte_test;
    $("#exp-entrenamiento").textContent =
      `Cohortes ${cohortesTrain || "—"} · validación ${d.cohorte_val ?? "—"} ` +
      `· test ${cohorteTest}→${cohorteTest + 1}.`;

    const nVars = (d.variables || []).length;
    $("#exp-modelo").textContent =
      `Gradient boosting calibrado, ${nVars} variable${nVars === 1 ? "" : "s"} ` +
      `(${d.modelo}).`;

    const lift10 = d.metricas_test.lift10;
    $("#exp-fiabilidad").textContent = lift10
      ? `Entre el 10 % de locales señalados como más arriesgados, cierran ` +
        `${lift10.toLocaleString("es-ES", { maximumFractionDigits: 1 })} veces ` +
        `más que la media. Es una estimación poblacional, no un diagnóstico ` +
        `sobre un negocio concreto.`
      : "—";

    // El simulador de local hipotético solo funciona si el campeón usa
    // variables derivables de un formulario. Si no (p. ej. el campeón F usa
    // 'n_locales_misma_marca', que necesita el padrón entero de Madrid),
    // se desactiva con la explicación en vez de dejar un formulario que
    // siempre falla con 501.
    if (d.predecir_disponible === false) {
      deshabilitarSimulador(d.predecir_faltan || []);
    }
  } catch (e) {
    console.error("No se pudo cargar /health", e);
  }
}

function deshabilitarSimulador(faltan) {
  const form = $("#form-simulador");
  if (form) {
    form.querySelectorAll("select, button").forEach((el) => (el.disabled = true));
  }
  const aviso = document.createElement("div");
  aviso.className = "mensaje-error";
  aviso.textContent =
    "El simulador no está disponible con el modelo actual: usa variables que " +
    "no se pueden calcular para un local que todavía no existe" +
    (faltan.length ? ` (${faltan.join(", ")})` : "") +
    ". Consulta un local real del censo en la sección 2.";
  $("#seccion-simulador").appendChild(aviso);
}

// ---------------------------------------------------------------------
// 1. Mapa: GET /barrios — color por QUINTILES (no lineal: el riesgo medio
// se mueve en un rango estrecho y una escala lineal sale casi toda amarilla)
// ---------------------------------------------------------------------
const PARADAS_COLOR = ["#ffffb2", "#fecc5c", "#fd8d3c", "#e31a1c", "#7f0000"];

function calcularQuintiles(valores) {
  const ord = [...valores].sort((a, b) => a - b);
  const corte = (p) => {
    const idx = (ord.length - 1) * p;
    const lo = Math.floor(idx), hi = Math.ceil(idx);
    return ord[lo] + (ord[hi] - ord[lo]) * (idx - lo);
  };
  // 4 cortes -> 5 tramos
  return [corte(0.2), corte(0.4), corte(0.6), corte(0.8)];
}

function tramoDe(v, cortes) {
  for (let i = 0; i < cortes.length; i++) {
    if (v <= cortes[i]) return i;
  }
  return cortes.length;
}

async function cargarMapa() {
  const mapa = L.map("mapa", { scrollWheelZoom: false }).setView([40.4168, -3.7038], 11);
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    attribution: "&copy; OpenStreetMap",
    maxZoom: 18,
  }).addTo(mapa);

  let barrios;
  try {
    const r = await fetch("/barrios");
    if (!r.ok) throw new Error("HTTP " + r.status);
    barrios = await r.json();
  } catch (e) {
    $("#seccion-mapa").insertAdjacentHTML(
      "beforeend",
      '<div class="mensaje-error">No se ha podido cargar el riesgo por barrio.</div>'
    );
    return [];
  }

  const riesgos = barrios.map((b) => b.riesgo_medio);
  const nlocales = barrios.map((b) => b.n_locales);
  const cortes = calcularQuintiles(riesgos);
  const minN = Math.min(...nlocales), maxN = Math.max(...nlocales);
  const radioPx = (n) => 5 + 25 * Math.sqrt((n - minN) / Math.max(1, maxN - minN));

  barrios.forEach((b) => {
    const tramo = tramoDe(b.riesgo_medio, cortes);
    const circulo = L.circleMarker([b.latitud, b.longitud], {
      radius: radioPx(b.n_locales),
      color: "#3a3a3a",
      weight: 1,
      fillColor: PARADAS_COLOR[tramo],
      fillOpacity: 0.88,
    }).addTo(mapa);
    circulo.bindTooltip(
      `<strong>${b.barrio}</strong> (${b.distrito})<br>` +
        `riesgo medio: ${pct(b.riesgo_medio)}<br>` +
        `locales: ${b.n_locales.toLocaleString("es-ES")}`
    );
  });

  // leyenda: 5 tramos con su rango real, ~26 barrios cada uno (quintiles)
  const bordes = [Math.min(...riesgos), ...cortes, Math.max(...riesgos)];
  const tramosHtml = PARADAS_COLOR.map((color, i) => {
    const desde = pct(bordes[i]);
    const hasta = pct(bordes[i + 1]);
    return `<span class="tramo-leyenda">
              <span class="muestra" style="background:${color}"></span>
              ${desde}–${hasta}
            </span>`;
  }).join("");
  $("#leyenda").innerHTML =
    `<div class="tramos">${tramosHtml}</div>` +
    `<div class="ayuda" style="margin-top:6px">5 tramos por quintiles del riesgo medio (~26 barrios cada uno) · tamaño = nº de locales</div>`;

  return barrios;
}

// ---------------------------------------------------------------------
// 2a. Filtro de barrio del buscador (reutiliza los barrios ya cargados)
// ---------------------------------------------------------------------
function poblarFiltroBarrio(barrios) {
  const sel = $("#sel-barrio-filtro");
  const nombres = [...new Set(barrios.map((b) => b.barrio))].sort((a, b) => a.localeCompare(b, "es"));
  for (const nombre of nombres) {
    const op = document.createElement("option");
    op.value = nombre;
    op.textContent = nombre;
    sel.appendChild(op);
  }
}

// ---------------------------------------------------------------------
// 2b. Buscador por nombre: GET /buscar?q=...&barrio=...  (con debounce)
// ---------------------------------------------------------------------
function ocultar(el) { el.hidden = true; el.innerHTML = ""; }

function pintarListaResultados(resultados) {
  const lista = $("#lista-resultados");
  const aviso = $("#aviso-busqueda");
  if (!resultados.length) {
    lista.hidden = true;
    aviso.hidden = false;
    aviso.textContent = "Sin resultados. Prueba con otro texto o quita el filtro de barrio.";
    return;
  }
  aviso.hidden = true;
  lista.hidden = false;
  lista.innerHTML = resultados
    .map(
      (r) => `
      <li data-id="${r.id_local}" tabindex="0">
        <span class="nombre">${r.rotulo}</span>
        <span class="detalle">${r.actividad}${r.barrio ? " · " + r.barrio : ""}</span>
        <span class="prob">${pct(r.probabilidad)}</span>
      </li>`
    )
    .join("");
  lista.querySelectorAll("li").forEach((li) => {
    const abrir = () => buscarPorId(li.dataset.id);
    li.addEventListener("click", abrir);
    li.addEventListener("keydown", (ev) => { if (ev.key === "Enter") abrir(); });
  });
}

async function ejecutarBusquedaNombre() {
  const q = $("#input-nombre").value.trim();
  const barrio = $("#sel-barrio-filtro").value;
  if (!q && !barrio) {
    ocultar($("#lista-resultados"));
    ocultar($("#aviso-busqueda"));
    return;
  }
  try {
    const params = new URLSearchParams({ q, barrio });
    const r = await fetch(`/buscar?${params.toString()}`);
    if (r.status === 404) {
      $("#aviso-busqueda").hidden = false;
      $("#aviso-busqueda").textContent = "El servidor no tiene este endpoint, reinícialo.";
      $("#lista-resultados").hidden = true;
      return;
    }
    if (!r.ok) throw new Error("HTTP " + r.status);
    pintarListaResultados(await r.json());
  } catch (e) {
    $("#aviso-busqueda").hidden = false;
    $("#aviso-busqueda").textContent = "No se ha podido buscar ahora mismo.";
    $("#lista-resultados").hidden = true;
  }
}

const ejecutarBusquedaNombreDebounced = debounce(ejecutarBusquedaNombre, 300);
$("#input-nombre").addEventListener("input", ejecutarBusquedaNombreDebounced);
$("#sel-barrio-filtro").addEventListener("change", ejecutarBusquedaNombre);

// ---------------------------------------------------------------------
// 2c. Ficha completa de un local: GET /local/{id_local}
// ---------------------------------------------------------------------
function pintarResultadoLocal(d) {
  const el = $("#resultado-buscador");
  el.innerHTML = `
    <div class="prob-grande">${pct(d.probabilidad)}</div>
    <div class="fila">
      <div class="dato"><span class="etiqueta">Rótulo</span><span class="valor">${d.rotulo}</span></div>
      <div class="dato"><span class="etiqueta">Actividad</span><span class="valor">${d.actividad}</span></div>
      <div class="dato"><span class="etiqueta">Distrito / barrio</span><span class="valor">${d.distrito}${d.barrio ? " · " + d.barrio : ""}</span></div>
      <div class="dato"><span class="etiqueta">Decil de riesgo</span><span class="valor">${d.decil_riesgo} / 10</span></div>
      <div class="dato"><span class="etiqueta">Tasa base (referencia)</span><span class="valor">${pct(d.tasa_base)}</span></div>
    </div>
    <p class="nota-poblacional">${d.nota}</p>
  `;
  el.hidden = false;
  el.scrollIntoView({ block: "nearest" });
}

async function buscarPorId(id) {
  const errorEl = $("#error-buscador");
  ocultar(errorEl);
  ocultar($("#resultado-buscador"));
  try {
    const r = await fetch(`/local/${encodeURIComponent(id)}`);
    if (r.status === 404) {
      errorEl.textContent = `No hay ningún local con id_local = "${id}" en el censo de junio de 2026.`;
      errorEl.hidden = false;
      return;
    }
    if (!r.ok) {
      errorEl.textContent = "El servicio no ha podido responder ahora mismo. Inténtalo de nuevo.";
      errorEl.hidden = false;
      return;
    }
    pintarResultadoLocal(await r.json());
  } catch (e) {
    errorEl.textContent = "No se ha podido contactar con la API.";
    errorEl.hidden = false;
  }
}

$("#form-buscador").addEventListener("submit", (ev) => {
  ev.preventDefault();
  const id = $("#input-id-local").value.trim();
  if (id) buscarPorId(id);
});

// ---------------------------------------------------------------------
// 3. Simulador: GET /catalogos + POST /predecir
// ---------------------------------------------------------------------
const MAX_OPCION = 60;

const esc = (s) =>
  String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

// Trunca a MAX_OPCION caracteres con "…"; el texto completo va en title.
function opcion(valor, textoCompleto) {
  const t = String(textoCompleto);
  const visible = t.length > MAX_OPCION ? t.slice(0, MAX_OPCION - 1).trimEnd() + "…" : t;
  return `<option value="${esc(valor)}" title="${esc(t)}">${esc(visible)}</option>`;
}

function llenarSelectSimple(select, valores) {
  const orden = [...valores].sort((a, b) => String(a).localeCompare(String(b), "es"));
  select.innerHTML = orden.map((v) => opcion(v, v)).join("");
}

function llenarSelectConDescripcion(select, items) {
  // Se ordenan aqui por descripcion (ademas de venir ya ordenados de la
  // API, por si acaso). Si algun item llega SIN descripcion (respuesta
  // antigua, o servidor no actualizado), no se pinta "undefined": se usa
  // el codigo como texto y se avisa por consola.
  const orden = [...items].sort((a, b) =>
    String(a.descripcion || a.codigo).localeCompare(String(b.descripcion || b.codigo), "es")
  );
  select.innerHTML = orden
    .map((it) => {
      const texto = it.descripcion || it.codigo;
      if (!it.descripcion) {
        console.warn(
          `/catalogos: el codigo '${it.codigo}' no trae 'descripcion'; ` +
            `mostrando el codigo tal cual. ¿API desactualizada?`
        );
      }
      return opcion(it.codigo, texto);
    })
    .join("");
}

async function cargarCatalogos() {
  try {
    const r = await fetch("/catalogos");
    if (r.status === 404) {
      $("#seccion-simulador").insertAdjacentHTML(
        "beforeend",
        '<div class="mensaje-error">El servidor no tiene este endpoint, reinícialo.</div>'
      );
      return;
    }
    if (!r.ok) throw new Error("HTTP " + r.status);
    const d = await r.json();
    llenarSelectConDescripcion($("#sel-epigrafe"), d.epigrafes || []);
    llenarSelectConDescripcion($("#sel-division"), d.divisiones || []);
    llenarSelectSimple($("#sel-distrito"), d.distritos || []);
    llenarSelectSimple($("#sel-acceso"), d.tipos_acceso || []);
  } catch (e) {
    console.error("No se pudo cargar /catalogos", e);
    $("#seccion-simulador").insertAdjacentHTML(
      "beforeend",
      '<div class="mensaje-error">No se han podido cargar los catálogos (epígrafe/distrito/...).</div>'
    );
  }
}

function pintarResultadoSimulador(d) {
  const el = $("#resultado-simulador");
  const veces = d.veces_sobre_la_media;
  const comparativa =
    veces >= 1
      ? `Riesgo ${veces.toLocaleString("es-ES", { maximumFractionDigits: 2 })} veces superior a la media de Madrid.`
      : `Riesgo ${(1 / veces).toLocaleString("es-ES", { maximumFractionDigits: 2 })} veces INFERIOR a la media de Madrid.`;
  const avisos = d.avisos.length
    ? `<ul class="avisos">${d.avisos.map((a) => `<li>${a}</li>`).join("")}</ul>`
    : "";
  el.innerHTML = `
    <div class="prob-grande">${pct(d.probabilidad)}</div>
    <div class="comparativa">${comparativa}</div>
    <div class="fila">
      <div class="dato"><span class="etiqueta">Decil de riesgo aprox.</span><span class="valor">${d.decil_riesgo_aprox} / 10</span></div>
      <div class="dato"><span class="etiqueta">Tasa base (referencia)</span><span class="valor">${pct(d.tasa_base)}</span></div>
    </div>
    ${avisos}
    <p class="nota-poblacional">${d.nota}</p>
  `;
  el.hidden = false;
}

$("#form-simulador").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const errorEl = $("#error-simulador");
  ocultar(errorEl);
  ocultar($("#resultado-simulador"));

  const payload = {
    epigrafe: $("#sel-epigrafe").value,
    division: $("#sel-division").value,
    distrito: $("#sel-distrito").value,
    tipo_acceso: $("#sel-acceso").value,
  };

  try {
    const r = await fetch("/predecir", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const cuerpo = await r.json();
    if (r.status === 422) {
      const msg = Array.isArray(cuerpo.detail)
        ? cuerpo.detail.map((e) => e.msg).join(" ")
        : cuerpo.detail;
      errorEl.textContent = msg;
      errorEl.hidden = false;
      return;
    }
    if (r.status === 501) {
      errorEl.textContent = cuerpo.detail;
      errorEl.hidden = false;
      return;
    }
    if (!r.ok) {
      errorEl.textContent = "El servicio no ha podido calcular el riesgo ahora mismo.";
      errorEl.hidden = false;
      return;
    }
    pintarResultadoSimulador(cuerpo);
  } catch (e) {
    errorEl.textContent = "No se ha podido contactar con la API.";
    errorEl.hidden = false;
  }
});

// ---------------------------------------------------------------------
cargarCabecera();
cargarMapa().then((barrios) => poblarFiltroBarrio(barrios || []));
cargarCatalogos();
