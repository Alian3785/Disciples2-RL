BX.ready(
        function()
        {

(function () {
  const CATEGORY_ID_TARGET = 9;

  // ✅ Поле, которое должно быть "1" при создании сделки
  const CREATE_FLAG_FIELD_ID = "UF_CRM_1702620262708";
  const CREATE_FLAG_FIELD_VAL = "1";

  // Для "второго риска" (copy=1)
  const SECOND_RISK_FLAG_KEY = "rec_second_risk_copy_flow";
  const SECOND_RISK_FIELD_ID = CREATE_FLAG_FIELD_ID;
  const SECOND_RISK_FIELD_VAL = CREATE_FLAG_FIELD_VAL;
  const SECOND_RISK_LIST_FIELD_ID = "UF_CRM_1555518235";
  const SECOND_RISK_SELECTED_VALUE_KEY = "rec_second_risk_selected_value";
  const SECOND_RISK_QUERY_PARAM = "second_risk_value";
  const SECOND_RISK_SOURCE_LINK_FIELD_ID = "UF_CRM_1772523775";
  const CROSS_SOURCE_LINK_FIELD_ID = "UF_CRM_1772525970";
  const CROSS_DUPLICATE_RESPONSIBLES_FIELD_ID = "UF_CRM_1773155052";

  // Для "кросса" (и ответственный для КОПИИ КОНТАКТА тоже)
  const RESPONSIBLE_ID_FOR_CROSS = 24209;

  // ------------------------- Общие хелперы -------------------------
  const TOP = window.top || window;
  const BXT = TOP.BX || window.BX;

  function notify(text) {
    const BX = BXT || window.BX;
    if (BX?.UI?.Notification?.Center) BX.UI.Notification.Center.notify({ content: text });
    else console.log(text);
  }

  function setFlowFlag(key) {
    try { sessionStorage.setItem(key, "1"); } catch (e) {}
  }
  function hasFlowFlag(key) {
    try { return sessionStorage.getItem(key) === "1"; } catch (e) { return false; }
  }
  function clearFlowFlag(key) {
    try { sessionStorage.removeItem(key); } catch (e) {}
  }
  function setFlowValue(key, value) {
    try {
      if (value == null) sessionStorage.removeItem(key);
      else sessionStorage.setItem(key, String(value));
    } catch (e) {}
  }
  function getFlowValue(key) {
    try {
      const v = sessionStorage.getItem(key);
      return v == null ? "" : String(v);
    } catch (e) {
      return "";
    }
  }

  // Берём category_id из URL (query) ИЛИ из пути вида /crm/deal/category/9/
  function getCategoryIdFromUrl(u) {
    try {
      const url = new URL(u, location.origin);

      const v = url.searchParams.get("category_id") || url.searchParams.get("CATEGORY_ID");
      if (v != null) {
        const n = parseInt(v, 10);
        if (Number.isFinite(n)) return n;
      }

      const m1 = (url.pathname || "").match(/\/crm\/deal\/category\/(\d+)\//i);
      if (m1) return parseInt(m1[1], 10);

      const m2 = (url.pathname || "").match(/\/crm\/deal\/category\/(\d+)(\/|$)/i);
      if (m2) return parseInt(m2[1], 10);

      return null;
    } catch (e) {
      return null;
    }
  }

  function getCategoryIdFromEditor(win) {
    try {
      const BX = win?.BX;
      const editor = BX?.Crm?.EntityEditor?.getDefault?.();
      const model = editor?.getModel ? editor.getModel() : editor?._model;
      if (!model) return null;

      let v = null;
      if (typeof model.getField === "function") v = model.getField("CATEGORY_ID");
      if (v == null && model._data && model._data.CATEGORY_ID != null) v = model._data.CATEGORY_ID;

      const n = v != null ? parseInt(v, 10) : NaN;
      return Number.isFinite(n) ? n : null;
    } catch (e) {
      return null;
    }
  }

  // true/false/undefined (если пока не удалось определить)
  function isCategory9Now(win) {
    const w = win || window;

    const byUrl =
      getCategoryIdFromUrl(w.location.href) ??
      getCategoryIdFromUrl((TOP.location && TOP.location.href) || "");
    if (byUrl != null) return byUrl === CATEGORY_ID_TARGET;

    const byEditor = getCategoryIdFromEditor(w) ?? getCategoryIdFromEditor(TOP);
    if (byEditor != null) return byEditor === CATEGORY_ID_TARGET;

    const BX = (w.BX || TOP.BX);
    const msgVal = BX?.message?.("CRM_DEAL_CATEGORY_ID") || BX?.message?.("CRM_ENTITY_CATEGORY_ID");
    if (msgVal != null) {
      const n = parseInt(msgVal, 10);
      if (Number.isFinite(n)) return n === CATEGORY_ID_TARGET;
    }

    return undefined;
  }

  function runOnlyIfCategory9(cb, opts) {
    const win = (opts && opts.win) || window;
    const MAX_TRIES = (opts && opts.maxTries) || 60;
    const INTERVAL = (opts && opts.interval) || 200;

    let t = 0;
    (function tick() {
      const ok = isCategory9Now(win);
      if (ok === true) return cb();
      if (ok === false) return;
      if (++t < MAX_TRIES) (win.setTimeout || setTimeout)(tick, INTERVAL);
    })();
  }

  function openDealById(dealId, editMode) {
    let url = "/crm/deal/details/" + parseInt(dealId, 10) + "/";
    if (editMode) url += "?init_mode=edit";
    const topWin = TOP || window;
    try {
      if (topWin.BX?.SidePanel?.Instance) topWin.BX.SidePanel.Instance.open(url, { cacheable: false });
      else topWin.location.href = url;
    } catch (e) {
      topWin.location.href = url;
    }
  }

  function getDealDetailsUrl(dealId) {
    const id = parseInt(dealId, 10);
    if (!id) return "";
    return (TOP?.location?.origin || location.origin) + "/crm/deal/details/" + id + "/";
  }

  function parseUrl(u) {
    try { return new URL(u, location.origin); } catch (e) { return null; }
  }

  // Универсальная установка значения поля в EntityEditor + "дожим" в контрол/DOM
  function setEntityEditorFieldValue(frameWindow, fieldId, value, opts) {
    const MAX_TRIES = opts?.maxTries ?? 80;
    const INTERVAL  = opts?.interval ?? 200;

    let tries = 0;
    (function apply() {
      if (!frameWindow) return;

      const FBX = frameWindow.BX;
      const editor = FBX?.Crm?.EntityEditor?.getDefault?.();

      if (!editor) {
        if (++tries < MAX_TRIES) frameWindow.setTimeout(apply, INTERVAL);
        return;
      }

      // 1) модель
      const model = editor.getModel ? editor.getModel() : editor._model;
      if (model?.setField) model.setField(fieldId, value);

      // 2) контрол
      const control = editor.getControlById ? editor.getControlById(fieldId, true) : null;
      if (control?.setValue) {
        control.setValue(value);
      } else {
        // 3) DOM fallback
        const doc = frameWindow.document;
        const input =
          doc.querySelector('[data-cid="' + fieldId + '"] input[type="text"]') ||
          doc.querySelector('[data-cid="' + fieldId + '"] textarea') ||
          doc.querySelector('input[name="' + fieldId + '"]') ||
          doc.querySelector('textarea[name="' + fieldId + '"]');

        if (input && input.value !== value) {
          input.value = value;
          try {
            input.dispatchEvent(new Event("input", { bubbles: true }));
            input.dispatchEvent(new Event("change", { bubbles: true }));
          } catch (e) {}
        }
      }

      // 4) несколько повторов чтобы пережить позднюю инициализацию контрола
      if (tries < 6) frameWindow.setTimeout(apply, INTERVAL);
    })();
  }

// =====================================================================
// 1) Переименование "Создать" -> "Создать рекомендацию"
// 2) Убираем (прячем) split-стрелку справа (button.ui-btn-menu) ТОЛЬКО у этой кнопки
// =====================================================================

(function () {
  // --- 0) Вставляем CSS один раз (чтобы "стрелка" не возвращалась) ---
  function ensureHideMenuCss() {
    if (document.getElementById("hide-create-reco-split-css")) return;

    const style = document.createElement("style");
    style.id = "hide-create-reco-split-css";
    style.textContent = `
      /* прячем именно помеченную нами кнопку-стрелку */
      button.ui-btn-menu[data-hide-create-reco-menu="1"]{
        display: none !important;
        width: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
        border: 0 !important;
      }
      /* возвращаем нормальные скругления основной кнопке */
      [data-create-reco-main="1"]{
        border-top-right-radius: 2px !important;
        border-bottom-right-radius: 2px !important;
      }
    `;
    document.head.appendChild(style);
  }

  // --- 1) Переименовываем кнопку ---
  function renameCreateButtonInDom(root) {
    root = root || document;

    const candidates = root.querySelectorAll(
      [
        ".ui-btn-main",
        ".ui-btn-success",
        ".ui-btn-primary",
        ".ui-btn.ui-btn-success",
        ".ui-btn.ui-btn-primary",
        ".crm-pagetitle-btn-box .ui-btn",
        ".pagetitle-menu .ui-btn",
        "button.ui-btn",
        "a.ui-btn"
      ].join(",")
    );

    let changed = false;

    candidates.forEach((btn) => {
      const textNode = btn.querySelector(".ui-btn-text") || btn;
      const raw = (textNode.textContent || "").replace(/\s+/g, " ").trim();
      const low = raw.toLowerCase();

      if (low === "создать") {
        textNode.textContent = "Создать рекомендацию";
        changed = true;
      }
    });

    return changed;
  }

  // --- утилита: ищем кнопку "Создать ..." в верхней панели ---
  function findCreateRecoButton(root) {
    root = root || document;

    const scope =
      root.querySelector(".crm-pagetitle-btn-box, .pagetitle-menu, .pagetitle-container") || root;

    const all = Array.from(scope.querySelectorAll("button, a, span"));
    return all.find((el) => {
      if (!(el instanceof HTMLElement)) return false;
      const tNode = el.querySelector?.(".ui-btn-text") || el;
      const txt = (tNode.textContent || "").replace(/\s+/g, " ").trim().toLowerCase();
      return (
        txt === "создать рекомендацию" ||
        txt === "создать рекоммендацию" ||
        txt === "создать"
      );
    }) || null;
  }

  // --- 2) Убираем split-стрелку справа (ui-btn-menu), привязанную к этой кнопке ---
  function hideMenuArrowNextToCreate(root) {
    root = root || document;
    ensureHideMenuCss();

    const createBtn = findCreateRecoButton(root);
    if (!createBtn) return false;

    // помечаем основную кнопку (для CSS-скруглений)
    createBtn.setAttribute("data-create-reco-main", "1");

    // Ищем ближайшую кнопку-стрелку ui-btn-menu, которая идёт ПОСЛЕ createBtn в DOM
    // 1) прямой сосед
    let menu = createBtn.nextElementSibling;
    if (menu && menu.matches && menu.matches("button.ui-btn-menu")) {
      menu.setAttribute("data-hide-create-reco-menu", "1");
      menu.style.display = "none";
      return true;
    }

    // 2) иногда стрелка — сосед родителя/обёртки
    let wrap = createBtn.parentElement;
    for (let i = 0; i < 8 && wrap; i++) {
      menu = wrap.nextElementSibling;
      if (menu && menu.matches && menu.matches("button.ui-btn-menu")) {
        menu.setAttribute("data-hide-create-reco-menu", "1");
        menu.style.display = "none";
        return true;
      }
      // 2b) или внутри этой же обёртки рядом
      menu = wrap.querySelector && wrap.querySelector("button.ui-btn-menu");
      if (menu) {
        // убедимся, что она относится именно к createBtn (после него по DOM)
        if (createBtn.compareDocumentPosition(menu) & Node.DOCUMENT_POSITION_FOLLOWING) {
          menu.setAttribute("data-hide-create-reco-menu", "1");
          menu.style.display = "none";
          return true;
        }
      }
      wrap = wrap.parentElement;
    }

    // 3) фолбэк: в пределах контейнера кнопок берём первую ui-btn-menu, которая стоит после createBtn
    let container = createBtn.parentElement;
    for (let i = 0; i < 10 && container; i++) {
      const menus = Array.from(container.querySelectorAll("button.ui-btn-menu"));
      const after = menus.find((m) => createBtn.compareDocumentPosition(m) & Node.DOCUMENT_POSITION_FOLLOWING);
      if (after) {
        after.setAttribute("data-hide-create-reco-menu", "1");
        after.style.display = "none";
        return true;
      }
      container = container.parentElement;
    }

    return false;
  }

  function ensureAll() {
    runOnlyIfCategory9(function () {
      let tries = 0;
      const max = 160;
      const interval = 200;

      (function loop() {
        const ok1 = renameCreateButtonInDom(document);
        const ok2 = hideMenuArrowNextToCreate(document);

        if (ok1 && ok2) return;
        if (++tries < max) setTimeout(loop, interval);
      })();
    }, { win: window, maxTries: 60, interval: 200 });
  }

  ensureAll();

  const debounced = (function () {
    let t;
    return function () {
      clearTimeout(t);
      t = setTimeout(ensureAll, 120);
    };
  })();

  if (window.MutationObserver) {
    new MutationObserver(debounced).observe(document.body, { childList: true, subtree: true });
  }
  window.addEventListener("popstate", debounced);
  window.addEventListener("hashchange", debounced);
})();


  // =====================================================================
  // 2) КНОПКА В СДЕЛКЕ (клон "Действия" + автозаполнение в форме создания)
  // =====================================================================
  runOnlyIfCategory9(function () {
    // === Автозаполнение: SOURCE_ID и UF-поля "Рекоммендация_ФИО" ===
    const TARGET_URL  = "https://crmnewcode.finist.com/crm/deal/details/0/?category_id=9";
    const SOURCE_CODE = "RECOMMENDATION";
    const FIELD_ID    = "UF_CRM_5B33779D7F3B7";
    const RECO_EXTRA_FIELD_ID = "UF_CRM_1772523775";
    const RECO_EXTRA_FIELD_VAL = "1";

    const B = (TOP.BX) ? TOP.BX : window.BX;

    function norm(u) {
      const a = document.createElement("a");
      a.href = u;
      return (a.pathname || "") + (a.search || "");
    }
    const TARGET_NORM = norm(TARGET_URL);

    function getCurrentUserName(cb) {
      if (B?.rest?.callMethod) {
        B.rest.callMethod("user.current")
          .then(function (res) {
            const u = res.data();
            const fio =
              [u.LAST_NAME, u.NAME, u.SECOND_NAME].filter(Boolean).join(" ").trim() ||
              u.NAME || u.LOGIN || String(u.ID || "");
            cb(fio);
          })
          .catch(function () { cb(""); });
      } else {
        const name = (B?.message && (B.message("USER_NAME") || B.message("USER_LOGIN"))) || "";
        cb(name || "");
      }
    }

    function setSource(frameWindow, tries) {
      const MAX_TRIES = 80, INTERVAL = 200;
      tries = tries || 0;
      if (!frameWindow) return;

      const FBX = frameWindow.BX;
      const editor = FBX?.Crm?.EntityEditor?.getDefault?.();

      if (!editor) {
        if (tries < MAX_TRIES) frameWindow.setTimeout(function () { setSource(frameWindow, tries + 1); }, INTERVAL);
        return;
      }

      const model = editor.getModel ? editor.getModel() : editor._model;
      if (model?.setField) model.setField("SOURCE_ID", SOURCE_CODE);

      const control = editor.getControlById ? editor.getControlById("SOURCE_ID", true) : null;
      if (control?.setValue) {
        control.setValue(SOURCE_CODE);
      } else {
        const hidden = frameWindow.document.querySelector('input[name="SOURCE_ID"]');
        if (hidden && hidden.value !== SOURCE_CODE) {
          hidden.value = SOURCE_CODE;
          try { hidden.dispatchEvent(new Event("change", { bubbles: true })); } catch (e) {}
        }
      }
    }

    function onSliderLoad(event) {
      const slider = event?.getTarget?.();
      const url = slider?.getUrl?.();
      if (!url) return;

      // создание (details/0) только для category_id=9
      if (norm(url) !== TARGET_NORM) return;

      const w = slider.getWindow ? slider.getWindow()
        : (slider.getFrameWindow ? slider.getFrameWindow() : null);
      if (!w) return;

      const ok = isCategory9Now(w);
      if (ok !== true) return;

      setSource(w, 0);

      // ✅ НОВОЕ: проставляем UF_CRM_1702620262708 = "1" в форме создания
      setEntityEditorFieldValue(w, CREATE_FLAG_FIELD_ID, CREATE_FLAG_FIELD_VAL, { maxTries: 80, interval: 200 });
      setEntityEditorFieldValue(w, RECO_EXTRA_FIELD_ID, RECO_EXTRA_FIELD_VAL, { maxTries: 80, interval: 200 });

      getCurrentUserName(function (fio) {
        const value = "Рекомендация_" + (fio || "");
        setEntityEditorFieldValue(w, FIELD_ID, value, { maxTries: 80, interval: 200 });
      });
    }

    B?.Event?.EventEmitter?.subscribe("SidePanel.Slider:onLoad", onSliderLoad);

    // ✅ НОВОЕ: если форма создания открыта НЕ в слайдере (полная страница)
    (function applyCreateFieldsOnFullPageIfNeeded() {
      try {
        if (norm(location.href) !== TARGET_NORM) return;
        runOnlyIfCategory9(function () {
          setSource(window, 0);
          setEntityEditorFieldValue(window, CREATE_FLAG_FIELD_ID, CREATE_FLAG_FIELD_VAL, { maxTries: 80, interval: 200 });
          setEntityEditorFieldValue(window, RECO_EXTRA_FIELD_ID, RECO_EXTRA_FIELD_VAL, { maxTries: 80, interval: 200 });

          getCurrentUserName(function (fio) {
            const value = "Рекомендация_" + (fio || "");
            setEntityEditorFieldValue(window, FIELD_ID, value, { maxTries: 80, interval: 200 });
          });
        }, { win: window, maxTries: 80, interval: 200 });
      } catch (e) {}
    })();

    // === Клон "Действия" ===
    const ACTIONS_CLONE_ID = "deal-actions-clone-btn";
    const ACTIONS_CLONE_TEXT = "Создать";
    const DEAL_DETAILS_NEEDLE = "crm/deal/details/";

    function isDealDetailsPage() {
      return location.href.indexOf(DEAL_DETAILS_NEEDLE) !== -1;
    }

    function findActionsBtn() {
      const nodes = document.querySelectorAll(
        ".pagetitle-menu button, .pagetitle-menu a," +
        ".crm-pagetitle-inner-container button, .crm-pagetitle-inner-container a," +
        ".crm-pagetitle-btn-box button, .crm-pagetitle-btn-box a"
      );
      for (const el of nodes) {
        const t = (el.textContent || "").replace(/\s+/g, " ").trim();
        if (/\u0414\u0435\u0439\u0441\u0442\u0432\u0438\u044f/i.test(t)) return el;
        if (/Действия/i.test(t)) return el;

        const hasPopupMenu = /^(true|menu)$/i.test(String(el.getAttribute("aria-haspopup") || ""));
        if (hasPopupMenu && /\bui-btn\b/.test(String(el.className || ""))) return el;
      }
      return null;
    }

    // ==========================================================
    // ✅ НОВОЕ: корректный якорь в TOP + "опустить" меню вниз
    // ==========================================================
    const CLONE_MENU_OFFSET_DOWN = 24; // px (регулируй)
    const CLONE_MENU_PROXY_ID = "deal-actions-clone-popup-anchor-proxy";

    function getRectInTopViewport(el) {
      const r = el.getBoundingClientRect();
      let left = r.left, top = r.top, width = r.width, height = r.height;

      let w = el.ownerDocument.defaultView;
      while (w && w !== w.top) {
        const frEl = w.frameElement;
        if (!frEl) break;
        const fr = frEl.getBoundingClientRect();
        left += fr.left;
        top  += fr.top;
        w = w.parent;
      }
      return { left, top, width, height };
    }

    function getBindElementInTop(anchor) {
      try {
        if (!anchor) return anchor;

        // Если уже в top-документе — ок
        if (anchor.ownerDocument === TOP.document) return anchor;

        // Иначе создаём "прокси-якорь" в TOP.document
        const doc = TOP.document;
        let proxy = doc.getElementById(CLONE_MENU_PROXY_ID);
        if (!proxy) {
          proxy = doc.createElement("span");
          proxy.id = CLONE_MENU_PROXY_ID;
          proxy.style.cssText = "position:absolute;width:1px;height:1px;opacity:0;pointer-events:none;";
          doc.body.appendChild(proxy);
        }

        const rr = getRectInTopViewport(anchor);
        const sx = TOP.pageXOffset || doc.documentElement.scrollLeft || 0;
        const sy = TOP.pageYOffset || doc.documentElement.scrollTop  || 0;

        // ставим прокси по центру кнопки, внизу
        proxy.style.left = Math.round(rr.left + rr.width / 2 + sx) + "px";
        proxy.style.top  = Math.round(rr.top + rr.height + sy) + "px";

        return proxy;
      } catch (e) {
        return anchor;
      }
    }

    function showCloneMenu(anchor) {
      if (!BXT?.PopupMenu) return;

      const opened = BXT.PopupMenu.getMenuById("deal-actions-clone-popup");
      if (opened && opened.popupWindow && opened.popupWindow.isShown()) { opened.close(); return; }

      const bindEl = getBindElementInTop(anchor);

      BXT.PopupMenu.show("deal-actions-clone-popup", bindEl, [
        { text: "Создать сделку для второго риска", onclick: function () { menuItem1_CreateSecondRisk(); this.popupWindow.close(); } },
        { text: "Создать кросс на другое направление", onclick: function () { this.popupWindow.close(); menuItem2_CreateCross(); } }
      ], {
        autoHide: true,
        cacheable: false,
        bindOptions: { position: "bottom" },
        angle: { position: "top" },
        offsetTop: CLONE_MENU_OFFSET_DOWN
      });
    }

    function ensureActionsClone() {
      if (!isDealDetailsPage()) return;

      runOnlyIfCategory9(function () {
        const actionsBtn = findActionsBtn();
        if (!actionsBtn || !actionsBtn.parentNode) return;

        let clone = document.getElementById(ACTIONS_CLONE_ID);
        if (!clone) {
          const tag = actionsBtn.tagName.toLowerCase() === "button" ? "button" : "a";
          clone = BXT.create(tag, {
            attrs: {
              id: ACTIONS_CLONE_ID,
              href: tag === "a" ? "#" : null,
              className: actionsBtn.className,
              style: "margin-right:8px;",
              title: ACTIONS_CLONE_TEXT,
              type: tag === "button" ? "button" : null
            },
            html: `<span>${ACTIONS_CLONE_TEXT}</span>`,
            events: { click: function (e) { e.preventDefault(); showCloneMenu(this); } }
          });
          clone.removeAttribute("aria-controls");
          clone.removeAttribute("aria-haspopup");
          clone.classList.remove("el-dropdown-selfdefine");
        }

        if (actionsBtn.previousElementSibling !== clone) {
          actionsBtn.parentNode.insertBefore(clone, actionsBtn);
        }
      }, { win: window, maxTries: 80, interval: 200 });
    }

    function debounceActions(fn, ms) { let t; return function () { clearTimeout(t); t = setTimeout(fn, ms); }; }
    const ensureActionsCloneDebounced = debounceActions(ensureActionsClone, 100);

    ensureActionsClone();
    BXT?.addCustomEvent?.("SidePanel.Slider:onOpenComplete", ensureActionsCloneDebounced);
    BXT?.addCustomEvent?.("SidePanel.Slider:onCloseComplete", ensureActionsCloneDebounced);
    if (window.MutationObserver) new MutationObserver(ensureActionsCloneDebounced).observe(document.body, { childList: true, subtree: true });
    window.addEventListener("popstate", ensureActionsCloneDebounced);
    window.addEventListener("hashchange", ensureActionsCloneDebounced);
  });

  // =====================================================================
  // 3) ПЕРВЫЙ ПУНКТ МЕНЮ: "Создать сделку для второго риска" (copy=1)
  // =====================================================================
  function openCurrentDealWithCopyParam(selectedRiskValue) {
    const m = location.pathname.match(/crm\/deal\/details\/(\d+)/);
    const id = m ? parseInt(m[1], 10) : null;
    if (!id) { notify("Не удалось определить ID сделки из URL"); return; }

    const params = new URLSearchParams();
    params.set("copy", "1");

    const riskVal = selectedRiskValue == null ? "" : String(selectedRiskValue).trim();
    if (riskVal) params.set(SECOND_RISK_QUERY_PARAM, riskVal);

    const url = `/crm/deal/details/${id}/?${params.toString()}`;
    const topWin = TOP || window;

    if (topWin.BX?.SidePanel?.Instance) {
      topWin.BX.SidePanel.Instance.open(url, { cacheable: false });
    } else {
      topWin.location.href = url;
    }
  }

  function normalizeSecondRiskOptions(raw) {
    const out = [];

    if (Array.isArray(raw)) {
      raw.forEach((item) => {
        if (!item) return;
        const id = item.ID ?? item.id ?? item.VALUE_ID ?? item.value_id ?? item.VALUE ?? item.value;
        const title = item.VALUE ?? item.value ?? item.NAME ?? item.name ?? id;
        if (id == null) return;
        out.push({ id: String(id), title: String(title ?? id) });
      });
    } else if (raw && typeof raw === "object") {
      Object.keys(raw).forEach((k) => {
        const item = raw[k];
        if (item && typeof item === "object") {
          const id = item.ID ?? item.id ?? item.VALUE_ID ?? item.value_id ?? k;
          const title = item.VALUE ?? item.value ?? item.NAME ?? item.name ?? id;
          out.push({ id: String(id), title: String(title ?? id) });
        } else if (item != null) {
          out.push({ id: String(k), title: String(item) });
        }
      });
    }

    const seen = new Set();
    return out.filter((x) => {
      if (!x || !x.id || x.id === "0") return false;
      if (seen.has(x.id)) return false;
      seen.add(x.id);
      return true;
    });
  }

  function getSecondRiskOptionsFromDom(win) {
    try {
      const doc = win?.document;
      if (!doc) return [];

      const select =
        doc.querySelector('[data-cid="' + SECOND_RISK_LIST_FIELD_ID + '"] select') ||
        doc.querySelector('select[name="' + SECOND_RISK_LIST_FIELD_ID + '"]');
      if (!select) return [];

      return Array.from(select.options || [])
        .map((o) => ({ id: String(o.value || ""), title: String((o.textContent || "").trim() || o.value || "") }))
        .filter((o) => o.id && o.id !== "0");
    } catch (e) {
      return [];
    }
  }

  function getCurrentSecondRiskValue(win) {
    try {
      const BX = win?.BX;
      const editor = BX?.Crm?.EntityEditor?.getDefault?.();
      const model = editor?.getModel ? editor.getModel() : editor?._model;
      if (!model) return "";

      let v = null;
      if (typeof model.getField === "function") v = model.getField(SECOND_RISK_LIST_FIELD_ID);
      if (v == null && model._data && model._data[SECOND_RISK_LIST_FIELD_ID] != null) v = model._data[SECOND_RISK_LIST_FIELD_ID];

      if (Array.isArray(v)) v = v[0];

      if (v && typeof v === "object") {
        const candidate =
          v.VALUE ?? v.value ??
          v.ID ?? v.id ??
          v.VALUE_ID ?? v.value_id ??
          v.ENUM_ID ?? v.enum_id;
        return candidate == null ? "" : String(candidate);
      }

      return v == null ? "" : String(v);
    } catch (e) {
      return "";
    }
  }

  async function loadSecondRiskOptions() {
    let options = [];

    if (restAvailable()) {
      try {
        const fields = await restCall("crm.deal.fields", {});
        const f = fields?.[SECOND_RISK_LIST_FIELD_ID];
        options = normalizeSecondRiskOptions(f?.items || f?.ITEMS || f?.LIST || f?.ENUM);
      } catch (e) {}
    }

    if (!options.length) options = getSecondRiskOptionsFromDom(window);
    if (!options.length) options = getSecondRiskOptionsFromDom(TOP);

    return options;
  }

  function startSecondRiskFlow(selectedRiskValue) {
    setFlowFlag(SECOND_RISK_FLAG_KEY);
    setFlowValue(SECOND_RISK_SELECTED_VALUE_KEY, selectedRiskValue);
    openCurrentDealWithCopyParam(selectedRiskValue);
  }

  async function updateCurrentDealSecondRiskBeforeCopy(selectedRiskValue) {
    const value = String(selectedRiskValue || "").trim();
    if (!value) throw new Error("�� ������� �������� �����.");

    const dealId = getDealIdFromUrl();
    if (!dealId) throw new Error("�� ������� ���������� ID ������� ������.");
    const sourceDealUrl = (TOP?.location?.origin || location.origin) + "/crm/deal/details/" + dealId + "/";

    // ��������� ������� ���� � ����� ����� (UI), � ����� ��������� �� �������.
    setEntityEditorFieldValue(window, SECOND_RISK_LIST_FIELD_ID, value, { maxTries: 40, interval: 100 });
    setEntityEditorFieldValue(window, SECOND_RISK_SOURCE_LINK_FIELD_ID, sourceDealUrl, { maxTries: 40, interval: 100 });

    if (!restAvailable()) {
      throw new Error("BX.rest.callMethod ����������, �� ���� ��������� ���� � �������� ������.");
    }

    await restCall("crm.deal.update", {
      id: dealId,
      fields: {
        [SECOND_RISK_LIST_FIELD_ID]: value,
        [SECOND_RISK_SOURCE_LINK_FIELD_ID]: sourceDealUrl
      }
    });
  }

  function showSecondRiskSelectModal() {
    const ok = isCategory9Now(window);
    if (ok !== true) { notify("Действие доступно только в направлении (category_id=9)."); return; }

    const BX = BXT || window.BX;
    if (!BX?.PopupWindowManager || !BX?.create) {
      notify("Не удалось открыть окно выбора риска.");
      return;
    }

    const POPUP_ID = "deal-second-risk-picker";
    const existing = BX.PopupWindowManager.getPopupById(POPUP_ID);
    if (existing) existing.destroy();

    let popup;
    let creating = false;
    const content = BX.create("div", { attrs: { style: "padding:16px; width: 440px;" } });
    content.appendChild(BX.create("div", {
      attrs: { style: "font-weight:600; margin-bottom:10px; font-size:15px;" },
      text: "Выберите риск"
    }));

    const sourceDealId = getDealIdFromUrl();
    const preselected = getCurrentSecondRiskValue(window) || getCurrentSecondRiskValue(TOP);
    const metaBox = BX.create("div", { attrs: { style: "font-size:12px; color:#5f6670; margin-bottom:10px; line-height:1.45;" } });
    const dealInfo = BX.create("div", { text: "ID копируемой сделки: " + (sourceDealId || "не определен") });
    const riskInfo = BX.create("div", {
      text: "Текущее значение " + SECOND_RISK_LIST_FIELD_ID + ": " + (preselected || "пусто")
    });
    metaBox.appendChild(dealInfo);
    metaBox.appendChild(riskInfo);
    content.appendChild(metaBox);

    const select = BX.create("select", {
      attrs: {
        style: "width:100%; min-height:36px; border:1px solid #cfd3d8; border-radius:4px; padding:6px 8px; margin-bottom:8px;"
      }
    });
    select.disabled = true;
    select.appendChild(BX.create("option", { attrs: { value: "" }, text: "Загрузка..." }));
    content.appendChild(select);

    const hint = BX.create("div", {
      attrs: { style: "font-size:12px; color:#8a9199;" },
      text: "После выбора будет открыта форма создания сделки для второго риска."
    });
    content.appendChild(hint);

    popup = BX.PopupWindowManager.create(POPUP_ID, null, {
      content: content,
      titleBar: "Создать сделку для второго риска",
      closeByEsc: true,
      closeIcon: { right: "12px", top: "10px" },
      lightShadow: true,
      overlay: { backgroundColor: "black", opacity: 50 },
      autoHide: false,
      draggable: { restrict: true },
      zIndex: 2300,
      minWidth: 440,
      buttons: [
        new BX.PopupWindowButton({
          text: "Создать",
          className: "ui-btn ui-btn-success",
          events: {
            click: function () {
              if (creating) return;
              const selected = String(select.value || "").trim();
              if (!selected) {
                notify("�������� ���� �� ������.");
                return;
              }

              creating = true;
              Promise.resolve()
                .then(function () { return updateCurrentDealSecondRiskBeforeCopy(selected); })
                .then(function () {
                  popup.close();
                  startSecondRiskFlow(selected);
                })
                .catch(function (e) {
                  creating = false;
                  notify("�� ������� ��������� ��������� ���� � �������� ������: " + (e?.message || e));
                });
            }
          }
        }),
        new BX.PopupWindowButton({
          text: "Отмена",
          className: "ui-btn ui-btn-link",
          events: { click: function () { popup.close(); } }
        })
      ]
    });

    popup.show();

    loadSecondRiskOptions()
      .then(function (options) {
        select.innerHTML = "";
        select.appendChild(BX.create("option", { attrs: { value: "" }, text: "Выберите значение..." }));

        options.forEach(function (opt) {
          const o = BX.create("option", { attrs: { value: opt.id }, text: opt.title });
          select.appendChild(o);
        });

        if (!options.length) {
          notify("Не удалось получить список значений поля " + SECOND_RISK_LIST_FIELD_ID);
          return;
        }

        if (preselected && options.some((x) => x.id === preselected)) {
          select.value = preselected;
        }
        if (preselected) {
          const selectedOption = options.find((x) => x.id === preselected);
          if (selectedOption) {
            riskInfo.textContent = "Текущее значение " + SECOND_RISK_LIST_FIELD_ID + ": " + preselected + " (" + selectedOption.title + ")";
          }
        }

        select.disabled = false;
      })
      .catch(function () {
        notify("Ошибка загрузки списка рисков.");
      });
  }

  function menuItem1_CreateSecondRisk() {
    const ok = isCategory9Now(window);
    if (ok !== true) { notify("Действие доступно только в направлении (category_id=9)."); return; }

    showSecondRiskSelectModal();
  }

  function tryApplySecondRiskFieldInWindow(w, sliderUrl) {
    const u = parseUrl(sliderUrl || w?.location?.href || "");
    const isCopy = !!(u && u.searchParams && u.searchParams.get("copy") === "1");
    if (!isCopy) return;
    const selectedFromUrl = String(u?.searchParams?.get(SECOND_RISK_QUERY_PARAM) || "").trim();
    const selectedRiskValue = selectedFromUrl || getFlowValue(SECOND_RISK_SELECTED_VALUE_KEY);
    const hasFlow = hasFlowFlag(SECOND_RISK_FLAG_KEY);
    if (!hasFlow && !selectedRiskValue) return;

    runOnlyIfCategory9(function () {
      // ✅ Теперь UF_CRM_1702620262708 = "1"
      setEntityEditorFieldValue(w, SECOND_RISK_FIELD_ID, SECOND_RISK_FIELD_VAL, { maxTries: 80, interval: 200 });
      if (selectedRiskValue) {
        setEntityEditorFieldValue(w, SECOND_RISK_LIST_FIELD_ID, selectedRiskValue, { maxTries: 80, interval: 200 });
      }
      clearFlowFlag(SECOND_RISK_FLAG_KEY);
      clearFlowFlag(SECOND_RISK_SELECTED_VALUE_KEY);
    }, { win: w, maxTries: 80, interval: 200 });
  }

  BXT?.Event?.EventEmitter?.subscribe?.("SidePanel.Slider:onLoad", function (event) {
    const slider = event?.getTarget?.();
    const url = slider?.getUrl?.();
    if (!url) return;

    const u = parseUrl(url);
    if (!u || u.searchParams.get("copy") !== "1") return;

    const w = slider.getWindow ? slider.getWindow()
      : (slider.getFrameWindow ? slider.getFrameWindow() : null);
    if (!w) return;

    tryApplySecondRiskFieldInWindow(w, url);
  });

  (function applySecondRiskOnFullPageIfNeeded() {
    const u = parseUrl(location.href);
    if (!u || u.searchParams.get("copy") !== "1") return;
    const selectedFromUrl = String(u.searchParams.get(SECOND_RISK_QUERY_PARAM) || "").trim();
    if (!hasFlowFlag(SECOND_RISK_FLAG_KEY) && !selectedFromUrl) return;

    tryApplySecondRiskFieldInWindow(window, location.href);
  })();

  // =====================================================================
  // 4) ВТОРОЙ ПУНКТ МЕНЮ: "Создать кросс на другое направление" (popup + AJAX)
  //    ✅ ДОРАБОТКА:
  //      - контакт копируется вместе с PHONE/EMAIL (и др. мультиполями)
  //      - ответственный у КОПИИ КОНТАКТА = 24209
  //      - ✅ НОВОЕ: в созданной сделке UF_CRM_1702620262708 = "1"
  // =====================================================================
  function menuItem2_CreateCross() {
    const ok = isCategory9Now(window);
    if (ok !== true) { notify("Действие доступно только в направлении (category_id=9)."); return; }
    resetCrossPhonePreviewCache();
    showOptionsModal();
  }

  const URL_WARP = "https://crmnewcode.finist.com/local/php_interface/js_libs/create_warp.php";
  const OPTIONS_10 = ["КАСКО+ОСАГО", "ИФЛ", "ИС (менеджер)"];
  const WARP_OPTION_TO_CATEGORY = { 1: 6, 2: 8, 3: 9 };

  function getCategoryIdByWarpOption(optionNumber) {
    const opt = parseInt(optionNumber, 10);
    if (!Number.isFinite(opt) || opt <= 0) return null;
    return WARP_OPTION_TO_CATEGORY[opt] || opt;
  }

  function getDealIdFromUrl() {
    const m = location.pathname.match(/crm\/deal\/details\/(\d+)/);
    return m ? parseInt(m[1], 10) : null;
  }

  // ------------------- ✅ НОВОЕ: копирование контакта -------------------
  const ENTITY_TYPE_CONTACT = 3; // Bitrix: CONTACT = 3
  const MULTIFIELD_KEYS = new Set(["PHONE", "EMAIL", "WEB", "IM"]);

  function restAvailable() {
    const BX = (TOP.BX || window.BX);
    return !!BX?.rest?.callMethod;
  }

  let restTimingSeq = 0;

  function nowMs() {
    if (typeof performance !== "undefined" && typeof performance.now === "function") {
      return performance.now();
    }
    return Date.now();
  }

  function shortStringify(value, maxLen) {
    let text = "";
    try {
      text = JSON.stringify(value);
    } catch (e) {
      text = "[unserializable]";
    }
    if (typeof text !== "string") text = String(text);
    if (text.length > maxLen) return text.slice(0, maxLen) + "...<truncated>";
    return text;
  }

  function summarizeRestParams(params) {
    if (!params || typeof params !== "object") return params;

    const out = {};
    if ("id" in params) out.id = params.id;
    if ("type" in params) out.type = params.type;
    if ("entity_type" in params) out.entity_type = params.entity_type;
    if ("values" in params) out.values = params.values;
    if ("filter" in params) out.filter = params.filter;
    if ("order" in params) out.order = params.order;
    if ("select" in params) out.select = params.select;
    if ("items" in params) out.itemsCount = Array.isArray(params.items) ? params.items.length : undefined;
    if ("fields" in params) {
      out.fieldsKeys = params.fields && typeof params.fields === "object"
        ? Object.keys(params.fields)
        : [];
    }
    if ("params" in params) out.params = params.params;

    return Object.keys(out).length ? out : params;
  }

  function beginRestTiming(mode, method, params) {
    const ctx = {
      id: ++restTimingSeq,
      mode: mode,
      method: method,
      startedAt: nowMs()
    };
    console.log(
      "[rest-timing][" + ctx.id + "] " + mode + " start " + method + " params=" + shortStringify(summarizeRestParams(params), 500)
    );
    return ctx;
  }

  function endRestTimingOk(ctx, extra) {
    const ms = Math.round((nowMs() - ctx.startedAt) * 100) / 100;
    const suffix = extra ? " " + extra : "";
    console.log("[rest-timing][" + ctx.id + "] " + ctx.mode + " ok " + ctx.method + " " + ms + "ms" + suffix);
  }

  function endRestTimingFail(ctx, errorText, extra) {
    const ms = Math.round((nowMs() - ctx.startedAt) * 100) / 100;
    const suffix = extra ? " " + extra : "";
    console.warn("[rest-timing][" + ctx.id + "] " + ctx.mode + " fail " + ctx.method + " " + ms + "ms error=" + (errorText || "unknown") + suffix);
  }

  function extractRestErrorText(err) {
    if (!err) return "unknown";
    if (typeof err === "string") return err;
    return err?.ex?.error_description || err?.error_description || err?.message || shortStringify(err, 250);
  }

  function restCall(method, params) {
    const BX = (TOP.BX || window.BX);
    return new Promise((resolve, reject) => {
      if (!BX?.rest?.callMethod) return reject(new Error("BX.rest.callMethod недоступен"));
      const timing = beginRestTiming("single", method, params);
      BX.rest.callMethod(method, params)
        .then((res) => {
          const data = res.data();
          const itemsCount = Array.isArray(data) ? data.length : (Array.isArray(data?.items) ? data.items.length : (Array.isArray(data?.result) ? data.result.length : null));
          endRestTimingOk(timing, itemsCount == null ? "" : "items=" + itemsCount);
          resolve(data);
        })
        .catch((err) => {
          endRestTimingFail(timing, extractRestErrorText(err));
          reject(err);
        });
    });
  }

  function restCallAll(method, params) {
    const BX = (TOP.BX || window.BX);
    return new Promise((resolve, reject) => {
      if (!BX?.rest?.callMethod) return reject(new Error("BX.rest.callMethod недоступен"));

      const timing = beginRestTiming("all", method, params || {});
      const allItems = [];
      let chunks = 0;
      let done = false;

      BX.rest.callMethod(method, params || {}, function onResult(result) {
        if (done) return;
        if (result.error()) {
          const err = result.error();
          const text = (typeof err === "string")
            ? err
            : (err?.ex?.error_description || err?.error_description || "REST error");
          done = true;
          endRestTimingFail(timing, text, "chunks=" + chunks + " items=" + allItems.length);
          reject(new Error(text));
          return;
        }

        const data = result.data();
        const chunk = normalizeListResult(data);
        chunks += 1;
        if (chunk.length) allItems.push(...chunk);

        if (typeof result.more === "function" && result.more()) {
          result.next();
          return;
        }

        done = true;
        endRestTimingOk(timing, "chunks=" + chunks + " items=" + allItems.length);
        resolve(allItems);
      });
    });
  }

  function normalizeListResult(data) {
    if (Array.isArray(data)) return data;
    if (data && Array.isArray(data.items)) return data.items;
    if (data && Array.isArray(data.result)) return data.result;
    return [];
  }

  function extractContactPhoneValues(contactRaw) {
    const contact = ensureMultifieldsFromFM(contactRaw || {});
    const rows = Array.isArray(contact?.PHONE)
      ? contact.PHONE
      : (contact?.PHONE && typeof contact.PHONE === "object" ? Object.values(contact.PHONE) : []);

    return rows
      .map((x) => (typeof x === "string" ? x : x?.VALUE))
      .map((x) => String(x || "").trim())
      .filter(Boolean);
  }

  function normalizePhone(value) {
    return String(value || "").replace(/\D+/g, "");
  }

  function buildPhoneKeys(value) {
    const digits = normalizePhone(value);
    if (!digits) return [];
    const keys = new Set([digits]);
    if (digits.length >= 10) keys.add(digits.slice(-10));
    return Array.from(keys);
  }

  function buildPhoneKeySet(phoneValues) {
    const set = new Set();
    (phoneValues || []).forEach((phone) => {
      buildPhoneKeys(phone).forEach((k) => set.add(k));
    });
    return set;
  }

  function intersectPhoneSets(a, b) {
    const out = [];
    if (!(a instanceof Set) || !(b instanceof Set) || !a.size || !b.size) return out;
    for (const k of a) if (b.has(k)) out.push(k);
    return out;
  }

  async function getContactPhoneKeySet(contactId, cache) {
    const id = parseInt(contactId, 10);
    if (!id) return new Set();
    if (cache && cache.has(id)) return cache.get(id);

    const contact = await restCall("crm.contact.get", { id: id });
    const phoneValues = extractContactPhoneValues(contact);
    const keys = buildPhoneKeySet(phoneValues);

    if (cache) cache.set(id, keys);
    return keys;
  }

  function contactDisplayName(c) {
    return [
      String(c?.LAST_NAME || "").trim(),
      String(c?.NAME || "").trim(),
      String(c?.SECOND_NAME || "").trim()
    ].filter(Boolean).join(" ");
  }

  function collectNumericIdsDeep(input, out) {
    if (!out) out = new Set();
    if (input == null) return out;
    if (Array.isArray(input)) {
      input.forEach((v) => collectNumericIdsDeep(v, out));
      return out;
    }
    if (typeof input === "object") {
      Object.keys(input).forEach((k) => {
        if (/id$/i.test(k)) {
          const n = parseInt(input[k], 10);
          if (Number.isFinite(n) && n > 0) out.add(n);
        }
        collectNumericIdsDeep(input[k], out);
      });
      return out;
    }
    const n = parseInt(input, 10);
    if (Number.isFinite(n) && n > 0) out.add(n);
    return out;
  }

  async function findContactIdsByPhoneViaDuplicate(sourcePhones) {
    const ids = new Set();
    if (!Array.isArray(sourcePhones) || !sourcePhones.length) return ids;

    for (const phone of sourcePhones) {
      try {
        const data = await restCall("crm.duplicate.findbycomm", {
          type: "PHONE",
          values: [phone],
          entity_type: "CONTACT"
        });
        collectNumericIdsDeep(data, ids);
      } catch (e) {
        // fallback ниже
      }
    }
    return ids;
  }

  async function getContactsByIds(ids) {
    const contactIds = Array.from(new Set((ids || [])
      .map((x) => parseInt(x, 10))
      .filter((x) => Number.isFinite(x) && x > 0)));

    if (!contactIds.length) return [];

    const wantedIds = new Set(contactIds);
    const byId = new Map();
    const chunkSize = 50;
    let hasBatchResponse = false;

    for (let i = 0; i < contactIds.length; i += chunkSize) {
      const chunk = contactIds.slice(i, i + chunkSize);
      try {
        const rows = await restCallAll("crm.contact.list", {
          filter: { ID: chunk },
          select: ["ID", "NAME", "LAST_NAME", "SECOND_NAME", "ASSIGNED_BY_ID", "PHONE", "EMAIL", "WEB", "IM", "FM"]
        });
        hasBatchResponse = true;
        (rows || []).forEach((row) => {
          const rowId = parseInt(row?.ID, 10);
          if (!rowId || !wantedIds.has(rowId) || byId.has(rowId)) return;
          byId.set(rowId, row);
        });
      } catch (e) {
        // fallback ниже
      }
    }

    const needFallbackIds = (!hasBatchResponse || byId.size < contactIds.length)
      ? contactIds.filter((id) => !byId.has(id))
      : [];

    if (needFallbackIds.length) {
      const fallbackRows = await Promise.all(needFallbackIds.map(async (id) => {
        try { return await restCall("crm.contact.get", { id: id }); } catch (e) { return null; }
      }));
      fallbackRows.filter(Boolean).forEach((row) => {
        const rowId = parseInt(row?.ID, 10);
        if (!rowId || byId.has(rowId)) return;
        byId.set(rowId, row);
      });
    }

    return Array.from(byId.values());
  }

  function normalizeDealRows(dealsRaw) {
    const byId = new Map();
    (Array.isArray(dealsRaw) ? dealsRaw : []).forEach((d) => {
      const dealId = parseInt(d?.ID, 10);
      if (!dealId || byId.has(dealId)) return;
      byId.set(dealId, {
        dealId: dealId,
        title: String(d?.TITLE || ""),
        stageId: String(d?.STAGE_ID || ""),
        categoryId: parseInt(d?.CATEGORY_ID, 10) || "",
        assignedById: parseInt(d?.ASSIGNED_BY_ID, 10) || "",
        closed: String(d?.CLOSED || "")
      });
    });
    return Array.from(byId.values()).sort((a, b) => Number(b.dealId) - Number(a.dealId));
  }

  async function getDealsByContactId(contactId) {
    const id = parseInt(contactId, 10);
    if (!id) return [];

    const queryParams = {
      select: ["ID", "TITLE", "STAGE_ID", "CATEGORY_ID", "ASSIGNED_BY_ID", "CLOSED"],
      order: { ID: "DESC" }
    };

    try {
      const primaryDeals = await restCallAll("crm.deal.list", Object.assign({}, queryParams, { filter: { CONTACT_ID: id } }));
      if (Array.isArray(primaryDeals) && primaryDeals.length) {
        return normalizeDealRows(primaryDeals);
      }
    } catch (e) {
      // fallback ниже
    }

    try {
      const fallbackDeals = await restCallAll("crm.deal.list", Object.assign({}, queryParams, { filter: { "=CONTACT_ID": id } }));
      return normalizeDealRows(fallbackDeals);
    } catch (e) {
      return [];
    }
  }

  async function findContactsBySourcePhones(sourceDealId) {
    const sourceContactId = await getDealPrimaryContactId(sourceDealId);
    if (!sourceContactId) return { sourceDealId: sourceDealId, sourceContactId: null, sourcePhones: [], contacts: [] };

    const sourceContact = await restCall("crm.contact.get", { id: sourceContactId });
    const sourcePhones = extractContactPhoneValues(sourceContact);
    const sourcePhoneKeys = buildPhoneKeySet(sourcePhones);
    if (!sourcePhoneKeys.size) {
      return { sourceDealId: sourceDealId, sourceContactId: sourceContactId, sourcePhones: sourcePhones, contacts: [] };
    }

    let candidateContacts = [];
    const idsByDuplicate = await findContactIdsByPhoneViaDuplicate(sourcePhones);
    if (idsByDuplicate.size) {
      candidateContacts = await getContactsByIds(Array.from(idsByDuplicate));
    }

    if (!candidateContacts.length) {
      candidateContacts = await restCallAll("crm.contact.list", {
        filter: { HAS_PHONE: "Y" },
        select: ["ID", "NAME", "LAST_NAME", "SECOND_NAME", "ASSIGNED_BY_ID", "PHONE"],
        order: { ID: "DESC" }
      });
    }

    const contacts = [];
    candidateContacts.forEach((c) => {
      const id = parseInt(c?.ID, 10);
      if (!id) return;

      const phones = extractContactPhoneValues(c);
      const phoneKeys = buildPhoneKeySet(phones);
      const matchedPhoneKeys = intersectPhoneSets(sourcePhoneKeys, phoneKeys);
      if (!matchedPhoneKeys.length) return;

      contacts.push({
        contactId: id,
        fio: contactDisplayName(c) || "—",
        assignedById: parseInt(c?.ASSIGNED_BY_ID, 10) || "",
        phones: phones,
        matchedPhones: matchedPhoneKeys,
        isSourceContact: id === sourceContactId,
        deals: []
      });
    });

    for (const c of contacts) {
      try {
        c.deals = await getDealsByContactId(c.contactId);
      } catch (e) {
        c.deals = [];
      }
    }

    return {
      sourceDealId: sourceDealId,
      sourceContactId: sourceContactId,
      sourcePhones: sourcePhones,
      contacts: contacts
    };
  }

  let crossPhonePreviewCache = { dealId: null, result: null };

  function resetCrossPhonePreviewCache() {
    crossPhonePreviewCache = { dealId: null, result: null };
  }

  function getEmptyCrossPhoneResult(sourceDealId) {
    return {
      sourceDealId: sourceDealId || null,
      sourceContactId: null,
      sourcePhones: [],
      contacts: []
    };
  }

  async function getCrossPhoneDetailsForDeal(sourceDealId) {
    const dealId = parseInt(sourceDealId, 10);
    if (!dealId) return getEmptyCrossPhoneResult(sourceDealId);

    if (crossPhonePreviewCache.dealId === dealId && crossPhonePreviewCache.result) {
      return crossPhonePreviewCache.result;
    }

    const result = await findContactsBySourcePhones(dealId);
    crossPhonePreviewCache = { dealId: dealId, result: result };
    return result;
  }

  const responsibleNameCache = new Map();

  function buildResponsibleName(user) {
    const name = String(user?.NAME || "").trim();
    const lastName = String(user?.LAST_NAME || "").trim();
    return [name, lastName].filter(Boolean).join(" ").trim();
  }

  async function getResponsibleNameById(assignedById) {
    const id = parseInt(assignedById, 10);
    if (!id) return "";

    if (responsibleNameCache.has(id)) {
      return responsibleNameCache.get(id);
    }

    let displayName = String(id);

    if (restAvailable()) {
      try {
        const raw = await restCall("user.get", { FILTER: { ID: id } });
        const users = Array.isArray(raw) ? raw : (Array.isArray(raw?.result) ? raw.result : []);
        const fio = buildResponsibleName(users[0] || null);
        if (fio) displayName = fio;
      } catch (e) {}
    }

    responsibleNameCache.set(id, displayName);
    return displayName;
  }

  async function enrichResponsibleNamesInCrossResult(result) {
    const contacts = Array.isArray(result?.contacts) ? result.contacts : [];
    const uniqueAssignedIds = new Set();

    contacts.forEach((c) => {
      const deals = Array.isArray(c?.deals) ? c.deals : [];
      deals.forEach((d) => {
        const assignedById = parseInt(d?.assignedById, 10);
        if (assignedById > 0) uniqueAssignedIds.add(assignedById);
      });
    });

    await Promise.all(Array.from(uniqueAssignedIds).map((id) => getResponsibleNameById(id)));

    contacts.forEach((c) => {
      const deals = Array.isArray(c?.deals) ? c.deals : [];
      deals.forEach((d) => {
        const assignedById = parseInt(d?.assignedById, 10);
        if (assignedById > 0) {
          d.assignedByName = responsibleNameCache.get(assignedById) || String(assignedById);
        } else {
          d.assignedByName = "";
        }
      });
    });

    return result;
  }

  const userInfoCache = new Map();
  const departmentHeadCache = new Map();
  const managerIdsByUserCache = new Map();

  function parseIdList(value) {
    if (value == null || value === "") return [];

    if (Array.isArray(value)) {
      return Array.from(new Set(value
        .map((x) => parseInt(x, 10))
        .filter((x) => Number.isFinite(x) && x > 0)));
    }

    if (typeof value === "string") {
      const text = value.trim();
      if (!text) return [];
      if (text.includes(",")) {
        return Array.from(new Set(text
          .split(",")
          .map((x) => parseInt(x.trim(), 10))
          .filter((x) => Number.isFinite(x) && x > 0)));
      }
      const n = parseInt(text, 10);
      return Number.isFinite(n) && n > 0 ? [n] : [];
    }

    if (typeof value === "object") {
      const out = [];
      Object.values(value).forEach((x) => {
        parseIdList(x).forEach((id) => out.push(id));
      });
      return Array.from(new Set(out));
    }

    const n = parseInt(value, 10);
    return Number.isFinite(n) && n > 0 ? [n] : [];
  }

  async function getUserById(userId) {
    const id = parseInt(userId, 10);
    if (!id || !restAvailable()) return null;

    if (userInfoCache.has(id)) return userInfoCache.get(id);

    let user = null;
    try {
      const raw = await restCall("user.get", { FILTER: { ID: id } });
      const rows = Array.isArray(raw) ? raw : normalizeListResult(raw);
      user = Array.isArray(rows) && rows.length ? rows[0] : null;
    } catch (e) {
      user = null;
    }

    userInfoCache.set(id, user);
    return user;
  }

  async function getDepartmentHeadIds(departmentId) {
    const depId = parseInt(departmentId, 10);
    if (!depId || !restAvailable()) return [];

    if (departmentHeadCache.has(depId)) return departmentHeadCache.get(depId);

    let headIds = [];
    try {
      const raw = await restCall("department.get", { ID: depId });
      const rows = Array.isArray(raw) ? raw : normalizeListResult(raw);
      const department = Array.isArray(rows) && rows.length ? rows[0] : null;
      if (department) {
        headIds = parseIdList(department?.UF_HEAD);
        if (!headIds.length) headIds = parseIdList(department?.HEAD);
      }
    } catch (e) {
      headIds = [];
    }

    const unique = Array.from(new Set(headIds.filter((id) => id > 0)));
    departmentHeadCache.set(depId, unique);
    return unique;
  }

  async function getManagerIdsForUser(userId) {
    const id = parseInt(userId, 10);
    if (!id) return [];

    if (managerIdsByUserCache.has(id)) return managerIdsByUserCache.get(id);

    const managerIds = new Set();
    const user = await getUserById(id);
    if (user) {
      parseIdList(user?.UF_HEAD).forEach((managerId) => {
        if (managerId > 0 && managerId !== id) managerIds.add(managerId);
      });
      parseIdList(user?.UF_MANAGER).forEach((managerId) => {
        if (managerId > 0 && managerId !== id) managerIds.add(managerId);
      });

      const departmentIds = parseIdList(user?.UF_DEPARTMENT);
      for (const depId of departmentIds) {
        const heads = await getDepartmentHeadIds(depId);
        heads.forEach((managerId) => {
          if (managerId > 0 && managerId !== id) managerIds.add(managerId);
        });
      }
    }

    const out = Array.from(managerIds);
    managerIdsByUserCache.set(id, out);
    return out;
  }

  async function collectRecipientsForSimpleNotification(deals) {
    const recipients = new Set();
    const responsibleIds = Array.from(new Set((Array.isArray(deals) ? deals : [])
      .map((deal) => parseInt(deal?.assignedById, 10))
      .filter((id) => Number.isFinite(id) && id > 0)));

    for (const responsibleId of responsibleIds) {
      recipients.add(responsibleId);
      const managerIds = await getManagerIdsForUser(responsibleId);
      managerIds.forEach((managerId) => {
        if (managerId > 0) recipients.add(managerId);
      });
    }

    const extraUserId = parseInt(RESPONSIBLE_ID_FOR_CROSS, 10);
    if (Number.isFinite(extraUserId) && extraUserId > 0) {
      recipients.add(extraUserId);
    }

    return Array.from(recipients);
  }

  /* legacy helpers removed
  function buildUserFio(user) {
    const lastName = String(user?.LAST_NAME || "").trim();
    const name = String(user?.NAME || "").trim();
    const secondName = String(user?.SECOND_NAME || "").trim();
    return [lastName, name, secondName].filter(Boolean).join(" ").trim();
  }

  function getCurrentUserId() {
    const BX = TOP.BX || window.BX;
    const id = parseInt(BX?.message?.("USER_ID"), 10);
    return Number.isFinite(id) && id > 0 ? id : 0;
  }

  function getCurrentUserFioFallback() {
    const BX = TOP.BX || window.BX;
    const fullName = String(BX?.message?.("USER_FULL_NAME") || "").trim();
    if (fullName) return fullName;

    const lastName = String(BX?.message?.("USER_LAST_NAME") || "").trim();
    const name = String(BX?.message?.("USER_NAME") || "").trim();
    const secondName = String(BX?.message?.("USER_SECOND_NAME") || "").trim();
    return [lastName, name, secondName].filter(Boolean).join(" ").trim();
  }

  async function getUserById(userId) {
    const id = parseInt(userId, 10);
    if (!id || !restAvailable()) return null;

    if (userInfoCache.has(id)) {
      return userInfoCache.get(id);
    }

    let user = null;
    try {
      const raw = await restCall("user.get", { FILTER: { ID: id } });
      const rows = Array.isArray(raw) ? raw : (Array.isArray(raw?.result) ? raw.result : normalizeListResult(raw));
      user = Array.isArray(rows) && rows.length ? rows[0] : null;
    } catch (e) {
      user = null;
    }

    userInfoCache.set(id, user);
    return user;
  }

  async function getDepartmentHeadIds(departmentId) {
    const depId = parseInt(departmentId, 10);
    if (!depId || !restAvailable()) return [];

    if (departmentHeadCache.has(depId)) {
      return departmentHeadCache.get(depId);
    }

    let headIds = [];
    try {
      const raw = await restCall("department.get", { ID: depId });
      const rows = Array.isArray(raw) ? raw : (Array.isArray(raw?.result) ? raw.result : normalizeListResult(raw));
      const department = Array.isArray(rows) && rows.length ? rows[0] : null;
      if (department) {
        headIds = parseIdList(department?.UF_HEAD);
        if (!headIds.length) headIds = parseIdList(department?.HEAD);
      }
    } catch (e) {
      headIds = [];
    }

    const unique = Array.from(new Set(headIds.filter((id) => id > 0)));
    departmentHeadCache.set(depId, unique);
    return unique;
  }

  async function getManagerIdsForUser(userId) {
    const id = parseInt(userId, 10);
    if (!id) return [];

    if (managerIdsByUserCache.has(id)) {
      return managerIdsByUserCache.get(id);
    }

    const managers = new Set();
    const user = await getUserById(id);
    if (user) {
      parseIdList(user?.UF_HEAD).forEach((managerId) => {
        if (managerId > 0 && managerId !== id) managers.add(managerId);
      });
      parseIdList(user?.UF_MANAGER).forEach((managerId) => {
        if (managerId > 0 && managerId !== id) managers.add(managerId);
      });

      const departmentIds = parseIdList(user?.UF_DEPARTMENT);
      for (const depId of departmentIds) {
        const headIds = await getDepartmentHeadIds(depId);
        headIds.forEach((managerId) => {
          if (managerId > 0 && managerId !== id) managers.add(managerId);
        });
      }
    }

    const list = Array.from(managers);
    managerIdsByUserCache.set(id, list);
    return list;
  }

  async function getDealCreatorFio(newDealId) {
    let creatorId = 0;

    if (restAvailable()) {
      try {
        const deal = await restCall("crm.deal.get", { id: newDealId });
        creatorId = parseInt(deal?.CREATED_BY_ID, 10) || 0;
      } catch (e) {}
    }

    if (!creatorId) {
      creatorId = getCurrentUserId();
    }

    if (creatorId) {
      const user = await getUserById(creatorId);
      const fio = buildUserFio(user);
      if (fio) return fio;
    }

    return getCurrentUserFioFallback() || "неизвестен";
  }

  */
  function normalizeDuplicateDealList(deals, sourceDealId) {
    return [];
    /*
    const sourceId = parseInt(sourceDealId, 10) || 0;
    const byId = new Map();

    (Array.isArray(deals) ? deals : []).forEach((deal) => {
      const dealId = parseInt(deal?.dealId ?? deal?.ID, 10);
      if (!dealId || dealId === sourceId || byId.has(dealId)) return;

      const assignedById = parseInt(deal?.assignedById ?? deal?.ASSIGNED_BY_ID, 10) || 0;
      if (!assignedById) return;

      byId.set(dealId, {
        dealId: dealId,
        title: String(deal?.title || deal?.TITLE || "").trim() || ("Сделка #" + dealId),
        assignedById: assignedById
      });
    });

    return Array.from(byId.values());
    */
  }

  async function resolveDuplicateDealsForNotification(sourceDealId, optionNumber, duplicateDealsHint) {
    return [];
    /*
    const hintedDeals = normalizeDuplicateDealList(duplicateDealsHint, sourceDealId);
    if (hintedDeals.length) return hintedDeals;

    const dealId = parseInt(sourceDealId, 10);
    if (!dealId) return [];

    try {
      const result = await getCrossPhoneDetailsForDeal(dealId);
      const filtered = filterCrossPhoneResultByOption(result, optionNumber);
      return collectDuplicateDealsForNotification(filtered, dealId);
    } catch (e) {
      console.warn("[cross-notify] Не удалось подготовить список дублей:", e);
      return [];
    }
    */
  }

  /* legacy helpers removed
  function sanitizeNotificationText(text) {
    return String(text == null ? "" : text)
      .replace(/\[/g, "(")
      .replace(/\]/g, ")")
      .replace(/\s+/g, " ")
      .trim();
  }

  function buildCrossDuplicateNotificationTextSafe(duplicateDeal, creatorFio) {
    const dealId = parseInt(duplicateDeal?.dealId, 10);
    const dealTitle = sanitizeNotificationText(
      duplicateDeal?.title || ("\u0421\u0434\u0435\u043b\u043a\u0430 #" + dealId)
    );
    const creatorName = sanitizeNotificationText(
      creatorFio || "\u043d\u0435\u0438\u0437\u0432\u0435\u0441\u0442\u0435\u043d"
    );
    const dealUrl = getDealDetailsUrl(dealId);
    const linkedTitle = "[URL=" + dealUrl + "]\u00ab" + dealTitle + "\u00bb[/URL]";
    return (
      "\u041f\u043e \u0432\u0430\u0448\u0435\u043c\u0443 \u043a\u043b\u0438\u0435\u043d\u0442\u0443 " +
      linkedTitle +
      " \u0441\u043e\u0437\u0434\u0430\u043d\u0430 \u0435\u0449\u0451 \u043e\u0434\u043d\u0430 \u0441\u0434\u0435\u043b\u043a\u0430. \u041c\u0435\u043d\u0435\u0434\u0436\u0435\u0440 \u2013 \u00ab" +
      creatorName +
      "\u00bb"
    );
  }

  function buildCrossDuplicateNotificationText(duplicateDeal, creatorFio) {
    const dealId = parseInt(duplicateDeal?.dealId, 10);
    const dealTitle = sanitizeNotificationText(duplicateDeal?.title || ("Сделка #" + dealId));
    const creatorName = sanitizeNotificationText(creatorFio || "неизвестен");
    const dealUrl = getDealDetailsUrl(dealId);
    const linkedTitle = "[URL=" + dealUrl + "]«" + dealTitle + "»[/URL]";
    return "По вашему клиенту " + linkedTitle + " создана ещё одна сделка. Менеджер – «" + creatorName + "»";
  }

  */
  async function sendWebhookNotification(userId, messageText) {
    return false;
    /*
    const id = parseInt(userId, 10);
    const message = String(messageText == null ? "" : messageText).trim();
    if (!id || !message) return false;

    const body = new URLSearchParams({
      to: String(id),
      message: message
    }).toString();

    try {
      await fetch(SIMPLE_DUPLICATE_WEBHOOK_URL, {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8" },
        body: body,
        credentials: "omit",
        mode: "cors"
      });
      return true;
    } catch (e) {
      try {
        await fetch(SIMPLE_DUPLICATE_WEBHOOK_URL, {
          method: "POST",
          headers: { "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8" },
          body: body,
          credentials: "omit",
          mode: "no-cors"
        });
        return true;
      } catch (e2) {
        console.warn("[cross-notify] Не удалось отправить webhook-уведомление:", e2);
        return false;
      }
    }
    */
  }

  async function notifyAboutCrossDuplicateDealsSimple(sourceDealId, optionNumber, duplicateDealsHint) {
    return;
    /*
    const deals = await resolveDuplicateDealsForNotification(sourceDealId, optionNumber, duplicateDealsHint);
    const duplicateDealIdsMessage = buildDuplicateDealIdsMessage(deals);
    const duplicateDealIds = duplicateDealIdsMessage
      .split(",")
      .map((id) => parseInt(id, 10))
      .filter((id) => Number.isFinite(id) && id > 0);
    console.log("[cross-notify] Duplicate deal IDs (force create):", duplicateDealIds);
    const targetUserId = parseInt(RESPONSIBLE_ID_FOR_CROSS, 10);
    const recipients = Number.isFinite(targetUserId) && targetUserId > 0 ? [targetUserId] : [];
    if (!recipients.length || !duplicateDealIdsMessage) return;

    await Promise.all(recipients.map(async (userId) => {
      try {
        await sendSystemNotification(userId, duplicateDealIdsMessage);
      } catch (e) {
        console.warn("[cross-notify] Не удалось отправить простое уведомление user=" + userId, e);
      }
    }));
    await Promise.all(recipients.map(async (userId) => {
      try {
        await sendWebhookNotification(userId, duplicateDealIdsMessage);
      } catch (e) {
        console.warn("[cross-notify] Не удалось отправить webhook-уведомление user=" + userId, e);
      }
    }));
    */
  }

  /* legacy helper removed
  async function notifyAboutCrossDuplicateDeals(newDealId, duplicateDeals) {
    const deals = normalizeDuplicateDealList(duplicateDeals, 0);
    if (!deals.length) return;

    const creatorFio = await getDealCreatorFio(newDealId);
    const sentPairs = new Set();
    let sentCount = 0;

    for (const deal of deals) {
      const responsibleId = parseInt(deal?.assignedById, 10);
      if (!responsibleId) continue;

      const recipients = new Set([responsibleId]);
      const managerIds = await getManagerIdsForUser(responsibleId);
      managerIds.forEach((managerId) => {
        if (managerId > 0) recipients.add(managerId);
      });

      const messageText = buildCrossDuplicateNotificationTextSafe(deal, creatorFio);
      for (const recipientId of recipients) {
        const pairKey = recipientId + ":" + deal.dealId;
        if (sentPairs.has(pairKey)) continue;
        sentPairs.add(pairKey);

        try {
          await sendSystemNotification(recipientId, messageText);
          sentCount += 1;
        } catch (e) {
          console.warn("[cross-notify] Ошибка отправки уведомления user=" + recipientId + " deal=" + deal.dealId, e);
        }
      }
    }

    console.log("[cross-notify] Уведомлений отправлено:", sentCount);
  }

  */
  function escapeHtml(text) {
    return String(text == null ? "" : text)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  async function notifyResponsibleIdsToCrossUser(deals) {
    const targetUserId = 25656;
    if (!targetUserId || !restAvailable()) return;

    const responsibleIds = Array.from(new Set((Array.isArray(deals) ? deals : [])
      .map((deal) => parseInt(deal?.assignedById, 10))
      .filter((id) => Number.isFinite(id) && id > 0)));
    if (!responsibleIds.length) return;

    await Promise.all(responsibleIds.map(async (responsibleId) => {
      const payload = {
        USER_ID: targetUserId,
        MESSAGE: String(responsibleId)
      };

      try {
        await restCall("im.notify.system.add", payload);
      } catch (e) {
        try {
          await restCall("im.notify.personal.add", payload);
        } catch (e2) {
          console.warn("[cross-notify] Failed to notify responsible id=" + responsibleId + " to user=" + targetUserId, e2);
        }
      }
    }));
  }

  function buildCrossPreviewHtml(result) {
    const contacts = Array.isArray(result?.contacts) ? result.contacts : [];
    const dealsMap = new Map();

    contacts.forEach((c) => {
      const deals = Array.isArray(c?.deals) ? c.deals : [];
      deals.forEach((d) => {
        const id = parseInt(d?.dealId, 10);
        if (!id || dealsMap.has(id)) return;
        dealsMap.set(id, {
          dealId: id,
          title: String(d?.title || "").trim() || "—",
          stageId: String(d?.stageId || "").trim() || "—",
          categoryId: (d?.categoryId == null || d?.categoryId === "") ? "—" : String(d.categoryId),
          assignedByName: String(d?.assignedByName || "").trim(),
          assignedById: (d?.assignedById == null || d?.assignedById === "") ? "—" : String(d.assignedById)
        });
      });
    });

    const deals = Array.from(dealsMap.values()).sort((a, b) => Number(b.dealId) - Number(a.dealId));

    let html = "";
    html += "<div style=\"font-weight:600; margin-bottom:6px;\">Найденные сделки</div>";
    html += "<div style=\"margin-bottom:8px;\">Сделок: <b>" + escapeHtml(deals.length) + "</b></div>";

    if (!deals.length) {
      html += "<div style=\"color:#7a828c;\">В выбранной воронке нет сделок в работе.</div>";
      return html;
    }

    deals.forEach((d) => {
      const responsibleText = d.assignedByName || d.assignedById;
      html += "<div style=\"border-top:1px solid #e3e7eb; padding-top:8px; margin-top:8px;\">";
      html += "<div><b>Сделка #" + escapeHtml(d.dealId) + "</b></div>";
      html += "<div>Название: " + escapeHtml(d.title) + "</div>";
      html += "<div>Ответственный: " + escapeHtml(responsibleText) + "</div>";
      html += "<div>Stage/Category: " + escapeHtml(d.stageId + "/" + d.categoryId) + "</div>";
      html += "</div>";
    });

    return html;
  }

  function collectCrossDuplicateResponsibleIds(result) {
    const contacts = Array.isArray(result?.contacts) ? result.contacts : [];
    const ids = new Set();

    contacts.forEach((contact) => {
      const deals = Array.isArray(contact?.deals) ? contact.deals : [];
      deals.forEach((deal) => {
        const responsibleId = parseInt(deal?.assignedById ?? deal?.ASSIGNED_BY_ID, 10);
        if (Number.isFinite(responsibleId) && responsibleId > 0) ids.add(responsibleId);
      });
    });

    return Array.from(ids).sort((a, b) => a - b);
  }

  async function saveCrossDuplicateResponsibleIdsToSourceDeal(sourceDealId, result) {
    const dealId = parseInt(sourceDealId, 10);
    if (!dealId || !restAvailable()) return;

    const responsibleIds = collectCrossDuplicateResponsibleIds(result);
    if (!responsibleIds.length) return;

    const value = responsibleIds.join(",");

    await restCall("crm.deal.update", {
      id: dealId,
      fields: {
        [CROSS_DUPLICATE_RESPONSIBLES_FIELD_ID]: value
      }
    });

    await restCall("crm.deal.add", {
      fields: {
        TITLE: String(value)
      }
    });



  }

  function isDealInWork(deal) {
    const closedRaw = String(deal?.closed || "").trim().toUpperCase();
    if (closedRaw) {
      return closedRaw !== "Y" && closedRaw !== "1" && closedRaw !== "TRUE";
    }

    // fallback, если CLOSED не пришёл
    const stage = String(deal?.stageId || "").trim().toUpperCase();
    return !/(WON|LOSE|FAIL|SUCCESS)/.test(stage);
  }

  function filterCrossPhoneResultByOption(result, optionNumber) {
    const targetCategoryId = getCategoryIdByWarpOption(optionNumber);
    if (!targetCategoryId) return result;

    const contactsRaw = Array.isArray(result?.contacts) ? result.contacts : [];
    const contacts = contactsRaw
      .map((c) => {
        const dealsRaw = Array.isArray(c?.deals) ? c.deals : [];
        const deals = dealsRaw.filter((d) => {
          const categoryId = parseInt(d?.categoryId, 10);
          return categoryId === targetCategoryId && isDealInWork(d);
        });
        return Object.assign({}, c, { deals: deals });
      })
      .filter((c) => Array.isArray(c.deals) && c.deals.length > 0);

    return {
      sourceDealId: result?.sourceDealId ?? null,
      sourceContactId: result?.sourceContactId ?? null,
      sourcePhones: Array.isArray(result?.sourcePhones) ? result.sourcePhones : [],
      contacts: contacts
    };
  }

  function logCrossPhoneContacts(result) {
    const sourceDealId = result?.sourceDealId ?? "-";
    const sourceContactId = result?.sourceContactId ?? "-";
    const sourcePhones = Array.isArray(result?.sourcePhones) ? result.sourcePhones : [];
    const contacts = Array.isArray(result?.contacts) ? result.contacts : [];
    const safeHeaderText = function (value) {
      return String(value == null ? "" : value).replace(/"/g, "'");
    };
    const formatDealIds = function (deals) {
      const ids = (Array.isArray(deals) ? deals : [])
        .map((d) => d?.dealId)
        .filter((id) => id != null && id !== "");
      return ids.length ? ids.join(", ") : "-";
    };
    const formatDealTitles = function (deals) {
      const titles = (Array.isArray(deals) ? deals : [])
        .map((d) => String(d?.title || "").trim() || "—");
      return titles.length ? titles.join(" | ") : "-";
    };
    const formatDealStageCategories = function (deals) {
      const rows = (Array.isArray(deals) ? deals : [])
        .map((d) => {
          const dealId = d?.dealId ?? "-";
          const stageId = String(d?.stageId || "-");
          const categoryId = d?.categoryId ?? "-";
          return "#" + dealId + "[" + stageId + "/" + categoryId + "]";
        });
      return rows.length ? rows.join(" | ") : "-";
    };
    const formatDealsShort = function (deals, maxItems) {
      const list = Array.isArray(deals) ? deals : [];
      const limit = Math.max(1, parseInt(maxItems, 10) || 3);
      const sliced = list.slice(0, limit);
      const text = sliced.map((d) => {
        const dealId = d?.dealId ?? "-";
        const stageId = d?.stageId || "-";
        const title = String(d?.title || "").trim() || "—";
        return "#" + dealId + " " + title + " [" + stageId + "]";
      }).join(" | ");
      return list.length > limit ? text + " | +" + (list.length - limit) + " ещё" : text;
    };

    const sourceContactRow = contacts.find((c) => String(c?.contactId ?? "") === String(sourceContactId));
    const sourceContactName = sourceContactRow?.fio ? safeHeaderText(sourceContactRow.fio) : "-";
    const title = "[cross-phone-contacts] sourceDeal=" + sourceDealId +
      " sourceContact=" + sourceContactId +
      " sourceContactName=\"" + sourceContactName + "\"" +
      " contacts=" + contacts.length;

    if (console.groupCollapsed) console.groupCollapsed(title);
    else console.log(title);

    console.log("sourcePhones:", sourcePhones);

    if (contacts.length) {
        if (console.table) {
          console.table(contacts.map((c) => ({
            contactId: c.contactId,
            fio: c.fio,
            assignedById: c.assignedById,
            isSourceContact: c.isSourceContact ? "Y" : "",
            phones: (c.phones || []).join(", "),
            matchedPhones: (c.matchedPhones || []).join(", "),
            dealsCount: Array.isArray(c.deals) ? c.deals.length : 0,
            dealIds: (Array.isArray(c.deals) ? c.deals : []).map((d) => d.dealId).join(", "),
            dealTitles: formatDealTitles(c.deals),
            dealsInfo: formatDealsShort(c.deals, 3)
          })));
        } else {
          console.log("contacts:", contacts);
        }
    } else {
      console.log("contacts: []");
    }

    contacts.forEach((c) => {
      const deals = Array.isArray(c.deals) ? c.deals : [];
      const contactName = safeHeaderText((String(c?.fio || "").trim() || "—"));
      const dealIds = formatDealIds(deals);
      const dealTitles = safeHeaderText(formatDealTitles(deals));
      const dealStageCategories = safeHeaderText(formatDealStageCategories(deals));
      const header = "[cross-phone-contacts] contact=" + c.contactId +
        " fio=\"" + contactName + "\"" +
        " deals=" + deals.length +
        " dealIds=" + dealIds +
        " dealTitles=\"" + dealTitles + "\"" +
        " dealStageCategory=\"" + dealStageCategories + "\"";
      console.log("[cross-phone-contacts-summary] contact=" + c.contactId +
        " fio=\"" + contactName + "\"" +
        " dealIds=" + dealIds +
        " dealTitles=\"" + dealTitles + "\"" +
        " dealStageCategory=\"" + dealStageCategories + "\"");
      if (console.groupCollapsed) console.groupCollapsed(header);
      else console.log(header);

      if (deals.length) {
        if (console.table) {
          console.table(deals.map((d) => ({
            dealId: d.dealId,
            title: d.title,
            categoryId: d.categoryId,
            stageId: d.stageId,
            assignedById: d.assignedById
          })));
        } else {
          console.log("deals:", deals);
        }
      } else {
        console.log("deals: []");
      }

      if (console.groupEnd) console.groupEnd();
    });

    if (console.groupEnd) console.groupEnd();
  }

  async function logDealContactPhones(dealId) {
    if (!restAvailable()) {
      console.log("[cross-phone] BX.rest недоступен, телефон контакта не прочитан");
      return;
    }

    try {
      const contactId = await getDealPrimaryContactId(dealId);
      if (!contactId) {
        console.log("[cross-phone] deal=" + dealId + " контакт не привязан");
        return;
      }

      const contact = await restCall("crm.contact.get", { id: contactId });
      const phones = extractContactPhoneValues(contact);

      const title = "[cross-phone] deal=" + dealId + " contact=" + contactId + " phones=" + phones.length;
      if (console.groupCollapsed) console.groupCollapsed(title);
      else console.log(title);
      console.log(phones);
      if (console.groupEnd) console.groupEnd();
    } catch (e) {
      console.log("[cross-phone] ошибка чтения телефонов:", e);
    }
  }

  function sanitizeMultiFieldArray(arr) {
    if (!Array.isArray(arr)) return arr;
    return arr
      .filter(x => x && (x.VALUE != null && String(x.VALUE).trim() !== ""))
      .map(x => ({
        VALUE: x.VALUE,
        VALUE_TYPE: x.VALUE_TYPE || x.TYPE || "WORK"
      }));
  }

  function ensureMultifieldsFromFM(contact) {
    // На всякий случай: если вдруг придёт FM вместо PHONE/EMAIL
    if (!contact || typeof contact !== "object") return contact;

    if (contact.FM && typeof contact.FM === "object") {
      if (!contact.PHONE && contact.FM.PHONE) contact.PHONE = contact.FM.PHONE;
      if (!contact.EMAIL && contact.FM.EMAIL) contact.EMAIL = contact.FM.EMAIL;
      if (!contact.WEB && contact.FM.WEB) contact.WEB = contact.FM.WEB;
      if (!contact.IM && contact.FM.IM) contact.IM = contact.FM.IM;
    }
    return contact;
  }

  async function getDealPrimaryContactId(dealId) {
    try {
      const data = await restCall("crm.deal.contact.items.get", { id: dealId });
      const items = normalizeListResult(data);
      if (!items.length) return null;

      const primary = items.find(x => x.IS_PRIMARY === "Y" || x.IS_PRIMARY === true || x.IS_PRIMARY === 1 || x.IS_PRIMARY === "1");
      return parseInt((primary || items[0]).CONTACT_ID, 10) || null;
    } catch (e) {
      try {
        const d = await restCall("crm.deal.get", { id: dealId });
        const cid = d?.CONTACT_ID;
        const n = cid != null ? parseInt(cid, 10) : NaN;
        return Number.isFinite(n) && n > 0 ? n : null;
      } catch (e2) {
        return null;
      }
    }
  }

  async function getWritableFieldsMapForContact() {
    const desc = await restCall("crm.contact.fields", {});
    return desc || {};
  }

  function pickWritableContactFields(contactRaw, fieldsDesc) {
    const contact = ensureMultifieldsFromFM(contactRaw || {});
    const out = {};
    const blacklist = new Set([
      "ID", "DATE_CREATE", "DATE_MODIFY", "CREATED_BY_ID", "MODIFY_BY_ID",
      "LAST_ACTIVITY_TIME", "LAST_ACTIVITY_BY", "HAS_PHONE", "HAS_EMAIL", "HAS_IM",
      "EXPORT", "ORIGIN_ID", "ORIGINATOR_ID"
    ]);

    Object.keys(contact || {}).forEach((k) => {
      if (blacklist.has(k)) return;

      const d = fieldsDesc ? fieldsDesc[k] : null;
      if (d && d.isReadOnly) return;

      let v = contact[k];
      if (v === undefined) return;

      // ✅ ВАЖНО: чистим мультиполя (PHONE/EMAIL/WEB/IM) от ID и лишних ключей
      if (MULTIFIELD_KEYS.has(k)) {
        v = sanitizeMultiFieldArray(v);
      }

      out[k] = v;
    });

    // На всякий случай: если в CONTACT поля лежат, но пустые — всё равно попробуем передать
    if (!out.PHONE && contact.PHONE) out.PHONE = sanitizeMultiFieldArray(contact.PHONE);
    if (!out.EMAIL && contact.EMAIL) out.EMAIL = sanitizeMultiFieldArray(contact.EMAIL);

    return out;
  }




  async function cloneContactFull(oldContactId) {
    const contact = await restCall("crm.contact.get", { id: oldContactId });
    const fieldsDesc = await getWritableFieldsMapForContact();
    const userId = +((top.BX || BX).message("USER_ID") || 0);

    const fields = pickWritableContactFields(contact, fieldsDesc);

    // ✅ Ответственный у КОПИИ контакта всегда 24209
    fields.ASSIGNED_BY_ID = RESPONSIBLE_ID_FOR_CROSS;
    fields.UF_CRM_618FEC5634E15 = [userId];


    // На всякий
    delete fields.ID;

    const newId = await restCall("crm.contact.add", {
      fields: fields,
      params: { REGISTER_SONET_EVENT: "N" }
    });

    return parseInt(newId, 10) || null;
  }

  async function copyRequisitesAndBankDetails(oldContactId, newContactId) {
    // best-effort: если реквизитов нет/модуль не используется — просто пропускаем
    try {
      const reqFieldsDesc = await restCall("crm.requisite.fields", {});
      const reqListData = await restCall("crm.requisite.list", {
        filter: { ENTITY_TYPE_ID: ENTITY_TYPE_CONTACT, ENTITY_ID: oldContactId }
      });
      const reqs = normalizeListResult(reqListData);
      if (!reqs.length) return;

      for (const req of reqs) {
        const oldReqId = parseInt(req.ID, 10);
        if (!oldReqId) continue;

        const reqFields = {};
        Object.keys(req || {}).forEach((k) => {
          if (k === "ID") return;
          const d = reqFieldsDesc ? reqFieldsDesc[k] : null;
          if (d && d.isReadOnly) return;
          reqFields[k] = req[k];
        });

        reqFields.ENTITY_TYPE_ID = ENTITY_TYPE_CONTACT;
        reqFields.ENTITY_ID = newContactId;

        const newReqId = await restCall("crm.requisite.add", { fields: reqFields });
        const newReqIdNum = parseInt(newReqId, 10);
        if (!newReqIdNum) continue;

        try {
          const bdFieldsDesc = await restCall("crm.requisite.bankdetail.fields", {});
          const bdListData = await restCall("crm.requisite.bankdetail.list", {
            filter: { ENTITY_ID: oldReqId }
          });
          const bds = normalizeListResult(bdListData);

          for (const bd of bds) {
            const bdFields = {};
            Object.keys(bd || {}).forEach((k) => {
              if (k === "ID") return;
              const d = bdFieldsDesc ? bdFieldsDesc[k] : null;
              if (d && d.isReadOnly) return;
              bdFields[k] = bd[k];
            });
            bdFields.ENTITY_ID = newReqIdNum;
            await restCall("crm.requisite.bankdetail.add", { fields: bdFields });
          }
        } catch (eBD) {
          // пропускаем
        }
      }
    } catch (e) {
      // пропускаем
    }
  }

  async function setDealSingleContact(dealId, contactId) {
    try {
      await restCall("crm.deal.contact.items.set", {
        id: dealId,
        items: [{ CONTACT_ID: contactId, IS_PRIMARY: "Y" }]
      });
      return true;
    } catch (e) {
      await restCall("crm.deal.update", {
        id: dealId,
        fields: { CONTACT_ID: contactId }
      });
      return true;
    }
  }

  async function cloneAndReplaceContactInNewDeal(oldDealId, newDealId) {
    if (!restAvailable()) {
      notify("Сделка создана, но BX.rest недоступен — контакт не скопирован.");
      return;
    }

function buildContactFio(c) {
  if (!c || typeof c !== "object") return "";
  return (
    (c.FULL_NAME && String(c.FULL_NAME).trim()) ||
    [c.LAST_NAME, c.NAME, c.SECOND_NAME].filter(Boolean).join(" ").trim()
  );
}

async function setDealTitle(dealId, title) {
  if (!restAvailable()) return;
  const t = (title || "").trim();
  if (!t) return;
  await restCall("crm.deal.update", {
    id: dealId,
    fields: { TITLE: t }
  });
}

async function setDealTitleFromContact(dealId, contactId) {
  if (!restAvailable()) return;
  const c = await restCall("crm.contact.get", { id: contactId });
  const fio = buildContactFio(c);
  if (fio) await setDealTitle(dealId, fio);
}

    const oldContactId = await getDealPrimaryContactId(oldDealId);
    if (!oldContactId) return; // нет контакта — ничего не делаем

    const newContactId = await cloneContactFull(oldContactId);
    if (!newContactId) throw new Error("Не удалось создать копию контакта");

    await copyRequisitesAndBankDetails(oldContactId, newContactId);
    await setDealSingleContact(newDealId, newContactId);
    await setDealTitleFromContact(newDealId, newContactId);

    notify("Контакт скопирован (включая PHONE/EMAIL) и заменён в новой сделке. Новый контакт: " + newContactId);
  }

  // ✅ НОВОЕ: проставить UF_CRM_1702620262708 = "1" уже созданной сделке (кросс)
  async function setCreateFlagOnExistingDeal(dealId, sourceDealId) {
    if (!restAvailable()) return;
    try {
      const fields = { [CREATE_FLAG_FIELD_ID]: CREATE_FLAG_FIELD_VAL };
      const srcId = parseInt(sourceDealId, 10);
      if (Number.isFinite(srcId) && srcId > 0) {
        fields[CROSS_SOURCE_LINK_FIELD_ID] = (TOP?.location?.origin || location.origin) + "/crm/deal/details/" + srcId + "/";
      }

      await restCall("crm.deal.update", {
        id: dealId,
        fields: fields
      });
    } catch (e) {
      console.error(e);
      notify("Не удалось обновить поля кросс-сделки: " + (e?.message || e));
    }
  }
  // ---------------------------------------------------------------------

  async function sendWarp(optionNumber) {
    const ok = isCategory9Now(window);
    if (ok !== true) { notify("Действие доступно только в направлении (category_id=9)."); return; }

    const BX = BXT || window.BX;
    const dealId = getDealIdFromUrl();
    if (!dealId) { notify("Не удалось определить ID текущей сделки"); return; }

    await logDealContactPhones(dealId);
    getCrossPhoneDetailsForDeal(dealId)
      .then(function (result) { logCrossPhoneContacts(filterCrossPhoneResultByOption(result, optionNumber)); })
      .catch(function (e) { console.log("[cross-phone-contacts] ошибка поиска контактов:", e); });

    BX.ajax({
      url: URL_WARP,
      method: "POST",
      dataType: "json",
      data: {
        sessid: BX.bitrix_sessid(),
        option: optionNumber,
        dealId: dealId,                 // чтобы PHP записал ссылку на исходную сделку
        assignedById: RESPONSIBLE_ID_FOR_CROSS
      },
      onsuccess: function (res) {
        if (res && res.status === "success") {
          if (res.id) {
            const newId = parseInt(res.id, 10);
            var e; if (false)
                console.warn("[cross-notify] Ошибка подготовки уведомлений:", e);
            notify("Создана сделка № " + newId + " — проставляю поле и проверяю контакт...");

            Promise.resolve()
              .then(function () { return setCreateFlagOnExistingDeal(newId, dealId); }) // ✅ UF_CRM_1702620262708="1" + ссылка на исходную сделку
              .then(function () { return cloneAndReplaceContactInNewDeal(dealId, newId); })
              .catch(function (e) {
                console.error(e);
                notify("Сделка создана, но контакт не удалось скопировать: " + (e?.message || e));
              })
              .finally(function () {
                notify("Открываю сделку № " + newId + " ...");
                openDealById(newId, true);
              });

          } else {
            notify("Копия создана, но ID не возвращён (сервер не передал id).");
          }
        } else {
          notify("Ошибка: " + (res && res.message ? res.message : "неизвестно"));
        }
      },
      onfailure: function () {
        notify("Ошибка запроса при создании кросса");
      }
    });
  }

  function showCrossConfirmModal(optionNumber, optionLabel) {
    const BX = BXT || window.BX;
    if (!BX?.PopupWindowManager || !BX?.create) return;

    const POPUP_ID = "deal-cross-confirm-create";
    const existing = BX.PopupWindowManager.getPopupById(POPUP_ID);
    if (existing) existing.destroy();

    let popup;
    let processing = false;

    const content = BX.create("div", { attrs: { style: "padding:16px; width: 420px;" } });
    content.appendChild(BX.create("div", {
      attrs: { style: "font-weight:600; margin-bottom:12px; font-size:16px;" },
      text: "Создать сделку?"
    }));
    const previewBox = BX.create("div", {
      attrs: {
        style: "margin-top:12px; max-height:260px; overflow:auto; border:1px solid #dfe4ea; border-radius:6px; padding:10px; font-size:12px; color:#2f3439; background:#fafbfc;"
      },
      text: "Загружаю найденные контакты и сделки..."
    });
    content.appendChild(previewBox);

    popup = BX.PopupWindowManager.create(POPUP_ID, null, {
      content: content,
      titleBar: "Подтверждение",
      closeByEsc: true,
      closeIcon: { right: "12px", top: "10px" },
      lightShadow: true,
      overlay: { backgroundColor: "black", opacity: 50 },
      autoHide: false,
      draggable: { restrict: true },
      zIndex: 2300,
      minWidth: 420,
      buttons: [
        new BX.PopupWindowButton({
          text: "Все равно создать",
          className: "ui-btn ui-btn-success",
          events: {
            click: function () {
              if (processing) return;
              processing = true;
              /*
                  console.warn("[cross-notify] Не удалось отправить упрощённые уведомления:", e);
                */
              popup.close();
              sendWarp(optionNumber);
            }
          }
        }),
        new BX.PopupWindowButton({
          text: "Отменить",
          className: "ui-btn ui-btn-link",
          events: { click: function () { popup.close(); } }
        })
      ]
    });

    popup.show();

    const dealId = getDealIdFromUrl();
    if (!dealId) {
      previewBox.textContent = "Не удалось определить ID текущей сделки.";
      return;
    }

    getCrossPhoneDetailsForDeal(dealId)
      .then(function (result) {
        const filtered = filterCrossPhoneResultByOption(result, optionNumber);
        return enrichResponsibleNamesInCrossResult(filtered);
      })
      .then(function (filtered) {
        if (!Array.isArray(filtered?.contacts) || !filtered.contacts.length) {
          previewBox.innerHTML =
            "<div style=\"font-weight:600; margin-bottom:6px;\">Найденные сделки</div>" +
            "<div>В выбранной воронке нет сделок в работе.</div>";
          return;
        }
        previewBox.innerHTML = buildCrossPreviewHtml(filtered);
        saveCrossDuplicateResponsibleIdsToSourceDeal(dealId, filtered)
          .catch(function (e) {
            console.warn("[cross-duplicates] Failed to save responsible IDs on source deal:", e);
          });
      })
      .catch(function (e) {
        previewBox.textContent = "Ошибка загрузки данных по контактам/сделкам: " + (e?.message || e);
      });
  }

  function showOptionsModal() {
    const ok = isCategory9Now(window);
    if (ok !== true) return;

    const BX = BXT || window.BX;
    if (!BX?.PopupWindowManager || !BX?.create) return;

    const POPUP_ID = "deal-actions-10-options";
    const existing = BX.PopupWindowManager.getPopupById(POPUP_ID);
    if (existing) { existing.show(); return; }

    let popup;
    const content = BX.create("div", { attrs: { style: "padding:12px 14px 14px; width:300px; box-sizing:border-box;" } });
    const list = BX.create("div", {
      attrs: {
        style: "display:grid; grid-template-columns:1fr; row-gap:6px; width:100%;"
      }
    });
    OPTIONS_10.forEach((label, idx) => {
      const optionNumber = idx + 1;
      const btn = BX.create("a", {
        attrs: {
          href: "#",
          className: "ui-btn ui-btn-light-border",
          style: "display:block; width:100%; margin:0; box-sizing:border-box; text-align:center;"
        },
        text: label,
        events: {
          click: function (e) {
            e.preventDefault();
            const clickedBtn = this;
            if (clickedBtn?.getAttribute?.("data-processing") === "1") return;

            if (clickedBtn?.setAttribute) clickedBtn.setAttribute("data-processing", "1");
            const oldText = clickedBtn && clickedBtn.textContent ? clickedBtn.textContent : label;
            if (clickedBtn) clickedBtn.textContent = "Поиск дублей";

            const dealId = getDealIdFromUrl();
            if (!dealId) {
              if (clickedBtn?.removeAttribute) clickedBtn.removeAttribute("data-processing");
              if (clickedBtn) clickedBtn.textContent = oldText;
              if (popup) popup.close();
              showCrossConfirmModal(optionNumber, label);
              return;
            }

            getCrossPhoneDetailsForDeal(dealId)
              .then(function (result) {
                const filtered = filterCrossPhoneResultByOption(result, optionNumber);
                const hasContacts = Array.isArray(filtered?.contacts) && filtered.contacts.length > 0;

                if (!hasContacts) {
                  if (popup) popup.close();
                  sendWarp(optionNumber);
                  return;
                }

                if (popup) popup.close();
                showCrossConfirmModal(optionNumber, label);
              })
              .catch(function () {
                if (popup) popup.close();
                showCrossConfirmModal(optionNumber, label);
              })
              .finally(function () {
                if (clickedBtn?.removeAttribute) clickedBtn.removeAttribute("data-processing");
                if (clickedBtn) clickedBtn.textContent = oldText;
              });
          }
        }
      });
      list.appendChild(btn);
    });
    content.appendChild(list);

    popup = BX.PopupWindowManager.create(POPUP_ID, null, {
      content: content,
      titleBar: "Выберете воронку",
      closeByEsc: true,
      closeIcon: { right: "10px", top: "10px" },
      lightShadow: true,
      overlay: { backgroundColor: "black", opacity: 50 },
      autoHide: true,
      draggable: { restrict: true },
      zIndex: 2200,
      minWidth: 300
    });

    popup.show();
  }
})();



urll = document.location.href;

const currentUsernewone = BX.message('USER_ID');

//Весь код ниже до условия с urll исправляет счетчик для пользователей Мекшило и Машкун. Для этого создается новая нода с правильным значением

			if (currentUsernewone == 33581 || currentUsernewone == 174) {

const elementByIdfull = document.getElementById('menu-counter-bp_tasks');

BX.rest.callMethod(
    'bizproc.task.list',
    {
        select: [
            'ID',
            'WORKFLOW_ID',
            'DOCUMENT_NAME',
            'DESCRIPTION',
            'NAME',
            'MODIFIED',
            'PARAMETERS'
        ],
        order: {
            ID: 'DESC'
        },
        filter: {
            'USER_ID': currentUsernewone,
            'STATUS': 0,
        }
    },
    function(result)
    {
        if(result.error())
            alert("Error: " + result.error());
        else
            console.log(result.data());
            const responseData3 = result.data();
            const taskCount2 = responseData3.length;
if (elementByIdfull) {
		if (currentUsernewone == 33581) {

const originalElement = document.querySelector('li[data-top-menu-id="top_menu_id_automation"] .menu-item-index-wrap');

// Вариант 2: Через href родительской ссылки (если он стабилен)
// const originalElement = document.querySelector('a[href="/bizproc/userprocesses/"] .menu-item-index-wrap');

// 2. Проверяем, найден ли оригинальный элемент
if (originalElement) {

    // 3. Создаем глубокую копию элемента (вместе со всем содержимым)
    const clonedElement = originalElement.cloneNode(true);

    // 4. Меняем класс у скопированного элемента
    // Если это единственный класс, можно просто присвоить новый:
    clonedElement.className = 'menu-item-index-wrap2';
    // Если могут быть другие классы, безопаснее использовать classList:

    // 5. Вставляем скопированный и измененный элемент в DOM
    // сразу после оригинального элемента
    originalElement.after(clonedElement);
    originalElement.style.display = 'none';

 const counterSpan = clonedElement.querySelector('span[data-role="counter"].menu-item-index');
    // Или можно использовать ID, если он точно скопировался и вы хотите его использовать:
    // const counterSpan = clonedElement.querySelector('#menu-counter-bp_tasks');

    // 3. Проверяем, найден ли внутренний span счетчика
    if (counterSpan) {

        // 4. Устанавливаем новое значение
        counterSpan.textContent = taskCount2-2; // Меняем видимый текст
        counterSpan.dataset.counterValue = taskCount2-2; // Меняем значение в data-атрибуте

		if (taskCount2 > 49) {counterSpan.textContent = "50+";}
    }

} else {
    // Сообщение об ошибке, если оригинальный элемент не найден
    console.error("Не удалось найти оригинальный элемент для копирования с помощью селектора.");
}
          //   elementByIdfull.textContent = taskCount2-2; // Устанавливаем новый текст
          //   elementByIdfull.dataset.counterValue = taskCount2-2;    
		}

	else {

const originalElement = document.querySelector('li[data-top-menu-id="top_menu_id_automation"] .menu-item-index-wrap');

// Вариант 2: Через href родительской ссылки (если он стабилен)
// const originalElement = document.querySelector('a[href="/bizproc/userprocesses/"] .menu-item-index-wrap');

// 2. Проверяем, найден ли оригинальный элемент
if (originalElement) {

    // 3. Создаем глубокую копию элемента (вместе со всем содержимым)
    const clonedElement = originalElement.cloneNode(true);

    // 4. Меняем класс у скопированного элемента
    // Если это единственный класс, можно просто присвоить новый:
    clonedElement.className = 'menu-item-index-wrap2';
    // Если могут быть другие классы, безопаснее использовать classList:

    // 5. Вставляем скопированный и измененный элемент в DOM
    // сразу после оригинального элемента
    originalElement.after(clonedElement);

    originalElement.style.display = 'none';

 const counterSpan = clonedElement.querySelector('span[data-role="counter"].menu-item-index');
    // Или можно использовать ID, если он точно скопировался и вы хотите его использовать:
    // const counterSpan = clonedElement.querySelector('#menu-counter-bp_tasks');

    // 3. Проверяем, найден ли внутренний span счетчика
    if (counterSpan) {

        counterSpan.textContent = taskCount2; // Меняем видимый текст
        counterSpan.dataset.counterValue = taskCount2; // Меняем значение в data-атрибуте

    }

} else {
    // Сообщение об ошибке, если оригинальный элемент не найден
    console.error("Не удалось найти оригинальный элемент для копирования с помощью селектора.");
}
          //   elementByIdfull.textContent = taskCount2-2; // Устанавливаем новый текст
          //   elementByIdfull.dataset.counterValue = taskCount2-2; 
}

}
    }
);

// Выводим найденный элемент в консоль для проверки (не обязательно)

}


			if (urll.includes("/bizproc/userprocesses/")) {

const currentUserId = BX.message('USER_ID');

				if (currentUserId == 25656 || currentUserId == 33581 || currentUserId == 174) {

const elementById = document.getElementById('menu-counter-bp_tasks');

// Выводим найденный элемент в консоль для проверки (не обязательно)

				}

//Исправляем счётчик на верные цифры  

const counterElement = document.querySelector('span.main-buttons-item-counter[data-mib-counter-id="top_menu_id_automation_menu_bizproc_sect_counter"]');

// Проверка, найден ли элемент и содержит ли он '6' (необязательно, но полезно)
if (counterElement) {

 // Может потребоваться BX.USER.GetID() в старых версиях или другом контексте

BX.rest.callMethod(
    'bizproc.task.list',
    {
        select: [
            'ID',
            'WORKFLOW_ID',
            'DOCUMENT_NAME',
            'DESCRIPTION',
            'NAME',
            'MODIFIED',
            'PARAMETERS'
        ],
        order: {
            ID: 'DESC'
        },
        filter: {
            'USER_ID': currentUserId,
            'STATUS': 0,
        }
    },
    function(result)
    {
        if(result.error())
            alert("Error: " + result.error());
        else
            console.log(result.data());
            const responseData2 = result.data();
            const taskCount = responseData2.length;
		if (currentUserId == 33581) {
            counterElement.textContent = taskCount - 2;
		}
		else {counterElement.textContent = taskCount;}

		if (taskCount > 49) {counterElement.textContent = "50+";}

    }
);


}}

			if (urll.includes("/tasks/task/view/")) {

//25725742745
var taskId = window.location.href.split("/").reverse()[1];


BX.rest.callMethod(
    'tasks.task.get', 
{taskId:taskId, select:['ID','TITLE', "GROUP_ID", "UF_CRM_TASK"]}, 
function(res){


var irtpid = res.answer.result["task"]["id"];

var ikkk = res.answer.result["task"]["title"];

var ikkkcrm = res.answer.result["task"]["ufCrmTask"][0];

var irtp = res.answer.result["task"]["group"]["id"];



	if (irtp == 45) {



BX.addCustomEvent('onPopupFirstShow', function(p) { 
        var menuId = 'task-view-b'; 
        if (p.uniquePopupId === 'menu-popup-' + menuId) 
        { 
            var menu = BX.PopupMenu.getMenuById(menuId), 
                href = window.location.href,  
                matches, taskId;

console.log(menu["menuItems"][0]["id"]);
var sothisidd = menu["menuItems"][3]["id"];
var sothisidd2 = menu["menuItems"][4]["id"];

menu.removeMenuItem(sothisidd);
menu.removeMenuItem(sothisidd2);


            if (matches = href.match(/\/task\/view\/([\d]+)\//i)) { 
                taskId = matches[1]; 
            } 
            //добавляем пункт меню, полученному по id  
        } 
    }); 

}


	if (ikkk.indexOf("Проверка  клиента") != -1) {


//селектор места вставки
        var completeButton = BX.findChild(//найти пасынков...
            BX('bx-component-scope-bitrix_tasks_widget_buttonstask_5'),//...для родителя
            {//с такими вот свойствами
                tag: 'span',
                className: 'task-view-button complete pause ui-btn ui-btn-success'
            },
            true//поиск рекурсивно от родителя
        );


	if (completeButton == null) {

        var completeButton = BX.findChild(//найти пасынков...
            BX('bx-component-scope-bitrix_tasks_widget_buttonstask_6'),//...для родителя
            {//с такими вот свойствами
                tag: 'span',
                className: 'task-view-button complete pause ui-btn ui-btn-success'
            },
            true//поиск рекурсивно от родителя
        );

}


var newButton2 = BX.create('span', {
           attrs: {                  
               className: 'task-view-button complete webform-small-button webform-small-button-accept'
           },
           text: 'Завершить задачу'
       });

BX.insertAfter(newButton2, completeButton);
BX.remove(completeButton);



BX.bind(newButton2, 'click', function(){


var btn_save = {

   title: BX.message('JS_CORE_WINDOW_SAVE'),

   id: 'savebtn',

   name: 'savebtn',

   className: BX.browser.IsIE() && BX.browser.IsDoctype() && !BX.browser.IsIE10() ? '' : 'adm-btn-save',

   action: function () {


var writee = document.getElementById("searchform");


writee[0].name = ikkkcrm;

writee[2].name = irtpid; 

writee[2].value = ikkk;


//document.getElementById("searchform").submit();



BX.ajax.submit(BX("searchform"));

this.parentWindow.Close();

BX.rest.callMethod(
	'tasks.task.complete', 
	{taskId:irtpid}, 
);

BX.remove(newButton2);

   }

};

//436734743274
//896585368658568
//22222222
//45786568568
//333333333333333333
//589568563856856856856
//700000080000000
//90000050000000
//3000070000000
//fffffffffffffffffffffffffff

var popup = new BX.CDialog({

   'title': "Заполнение полей",

   'content': "<form method='POST' style='overflow:hidden;' action='https://crm.finist.com//local/php_interface/js_libs/phppage.php' id='searchform'>\
        <label for=search>ВИН ТС</label>\
        <textarea name='search' style='height: 40px; width: 600px;'></textarea>\
        <label for=toosearch>Решение андера по проверке клиента</label>\
        <select name='toosearch' style='height: 35px; width: 150px;'> \
<option value='73608'>Берем через наш АД</option> \
<option value='73605'>Берем через субАД</option> \
<option value='73611'>Запрет на страхование</option> </select>\
        <input type='hidden' name='hidden_field_name' value='hidden_value'>\
        <label for=commen>Комментарий</label>\
        <textarea name='commen' style='height: 25px; width: 150px;'></textarea>\
        </form>",

   'draggable': true,

   'resizable': true,

   'buttons': [btn_save, BX.CDialog.btnCancel]

});


popup.Show();


});

};



    if (ikkk.indexOf("Подготовить проект по сделке") != -1) {

var erttttt = ikkkcrm.replace(/[^0-9]/g, '');

BX.rest.callMethod(
    'crm.deal.get', 
{id:erttttt,}, 
function(res2){

tunzz = res2.answer.result;


var soletsadd = tunzz["UF_CRM_1724335598907"];
var soletsadd2 = tunzz["UF_CRM_1724336577"];
var soletsadd3 = tunzz["UF_CRM_1724354420"];
var soletsadd4 = tunzz["UF_CRM_WEB_TWITTER"];
var soletsadd5 = tunzz["UF_CRM_IM_INSTAGRAM"];

//селектор места вставки
        var completeButton = BX.findChild(//найти пасынков...
            BX('bx-component-scope-bitrix_tasks_widget_buttonstask_5'),//...для родителя
            {//с такими вот свойствами
                tag: 'span',
                className: 'task-view-button complete pause ui-btn ui-btn-success'
            },
            true//поиск рекурсивно от родителя
        );


	if (completeButton == null) {

        var completeButton = BX.findChild(//найти пасынков...
            BX('bx-component-scope-bitrix_tasks_widget_buttonstask_6'),//...для родителя
            {//с такими вот свойствами
                tag: 'span',
                className: 'task-view-button complete pause ui-btn ui-btn-success'
            },
            true//поиск рекурсивно от родителя
        );

}



        var newButton2 = BX.create('span', {
            attrs: {
                className: 'task-view-button complete webform-small-button webform-small-button-accept'
            },
            text: 'Завершить задачу'
        });

        BX.insertAfter(newButton2, completeButton);
        BX.remove(completeButton);



        BX.bind(newButton2, 'click', function(){


            var btn_save = {

                title: BX.message('JS_CORE_WINDOW_SAVE'),

                id: 'savebtn',

                name: 'savebtn',

                className: BX.browser.IsIE() && BX.browser.IsDoctype() && !BX.browser.IsIE10() ? '' : 'adm-btn-save',

                action: function () {


                    var writee = document.getElementById("searchform3");

                    console.log(writee);


                    writee[0].name = ikkkcrm;


                    //document.getElementById("searchform3").submit();

                    BX.ajax.submit(BX("searchform3"));

                    this.parentWindow.Close();

                    BX.rest.callMethod(
                        'tasks.task.complete',
                        {taskId:irtpid},
                    );

                    BX.remove(newButton2);

                }

            };



            var popup = new BX.CDialog({

                'title': "Наберите данные",

                'content': "<form method='POST' style='overflow:hidden;' action='https://crm.finist.com//local/php_interface/js_libs/phppage3.php' id='searchform3'>\
        <label for=search>Скидка СК от андера:</label>\
        <textarea id='search' name='search' style='height: 50px; width: 500px;'></textarea>\
        <label for=search2>Скидка из КВ ОФ от андера:</label>\
        <textarea id='search2' name='search2' style='height: 50px; width: 450px;'></textarea>\
        <label for=search3>Скидка из КВ не ОФ от андера:</label>\
        <textarea id='search3' name='search3' style='height: 50px; width: 450px;'></textarea>\
		<label for=verynew>Страховая компания</label>\
        <select id='search4' name='verynew' style='height: 35px; width: 150px;'> \
		<option value='Ингосстрах'>Ингосстрах</option> \
		<option value='Альфастрахование'>Альфастрахование</option> \
		<option value='ВСК'>ВСК</option> \
		<option value='Ренессанс'>Ренессанс</option> \
		<option value='ЮГОРИЯ'>ЮГОРИЯ</option> \
		<option value='Пари'>Пари</option> \
		<option value='РЕСО'>РЕСО</option> \
		<option value='Сберстрахование'>Сберстрахование</option> \
		<option value='Энергогарант'>Энергогарант</option> \
		<option value='Росгосстрах'>Росгосстрах</option> \
		<option value='ОСК'>ОСК</option> \
		<option value='Зетта'>Зетта</option> \
		<option value='Совкомбанк Страхование'>Совкомбанк Страхование</option> \
		<option value='Астро-Волга'>Астро-Волга</option> \
		<option value='АМТ Страхование'>АМТ Страхование</option> \
		<option value='Инсайт'>Инсайт</option> </select>\
        <input type='hidden' name='hiddenField' id='hiddenField' value='' />\
        </form>",

                'draggable': true,

                'resizable': true,

                'buttons': [btn_save, BX.CDialog.btnCancel]

            });

 //var xx = soletsadd;
 //var xx2 = soletsadd2;
 //var xx3 = soletsadd3;
 //var xx4 = soletsadd4;
 //var xx5 = soletsadd5;

  // Replace with your dynamic value
 document.getElementById('search').value = soletsadd;
 document.getElementById('search2').value = soletsadd2;
 document.getElementById('search3').value = soletsadd3;
 document.getElementById('search4').value = soletsadd4;
 document.getElementById('hiddenField').value = soletsadd5;

            popup.Show();


        });

});


//Дальше код новых оценок

var sidebarMarkElement = BX.findChild(//найти пасынков... 
		BX('tasks-iframe-popup-scope'),//...для родителя 
			{//с такими вот свойствами 
		className: 'task-detail-sidebar-item task-detail-sidebar-item-mark' 
			}, 
			true//поиск рекурсивно от родителя 
			); 

	if (sidebarMarkElement)  
	{


		var currentUrl = window.location.href, matches, taskId; 
		//узнаем id задачи из URL 
		if (matches = currentUrl.match(/\/task\/view\/([\d]+)\//i)) { 
			taskId = matches[1]; 
	} 

document.querySelectorAll('.task-detail-sidebar-item-title')
        .forEach(n => n.textContent.trim() === 'Оценка:' && (n.textContent = 'Оценка менеджера:'));

	//создаем кнопку 
 var containerofmark = BX.create('div', {
        attrs: {
            // можно добавить стили, если нужно, но по умолчанию inline-элементы внутри div будут в строку
        }
    });


    // создаём span с надписью "Оценка андера:"
    var labelSpan = BX.create('span', {
        text: 'Оценка андера',
    });

    // создаём span с классом justsome и текстом "Нет оценки"
    var ratingSpan = BX.create('span', {
        attrs: {
            className: 'justsome',
            style: 'margin-left: 12px; text-decoration: underline dashed;'
        },
        text: 'Нет оценки'
    });

    // вкладываем оба span в контейнер
    BX.append(labelSpan, containerofmark);
    BX.append(ratingSpan, containerofmark);

    // вставляем контейнер сразу после sidebarMarkElement
    BX.insertAfter(containerofmark, sidebarMarkElement);

fetch('https://crm.finist.com/filetogetmark.php', {
                                                        method: 'post',
                                                        body: JSON.stringify({
                                                                a: taskId,
                                                                b: "P"
                                                        })
                                                }) .then(function(response) {
                                                        if (response.status >= 200 && response.status < 300) {
                                                                return response.text();
																//rtttt7 = response.text();
																//console.log(rtttt7);
                                                        }
                                                        throw new Error(response.statusText)
                                                })
                                                    .then(function(response) {
																console.log(response);
														if (response == "P") {
													var ratingDiv = document.querySelector('span.justsome');
													if (ratingDiv) {
														ratingDiv.textContent = 'Положительная';
ratingDiv.style.color = 'green';
 ratingDiv.style.textDecoration = 'underline dashed';
													}
}

														if (response == "N") {
													var ratingDiv = document.querySelector('span.justsome');
													if (ratingDiv) {
														ratingDiv.textContent = 'Отрицательная';
ratingDiv.style.color = 'red';
ratingDiv.style.textDecoration = 'underline dashed';
													}
}


                                                    })

BX.bind(ratingSpan, 'click', function(){


        var ratingPopup = BX.PopupWindowManager.create("popup-message", BX('task-detail-mark'), {

                content: ''
            + '<div class="task-grade-popup-title">Оценка андера</div>'
            + '<div class="task-popup-list-list">'
            +   '<a class="task-popup-list-item">'
            +     '<span class="task-popup-list-item-left"></span>'
            +     '<span class="task-popup-list-item-icon task-popup-grade-icon-none"></span>'
            +     '<span class="task-popup-list-item-text">Нет оценки</span>'
            +     '<span class="task-popup-list-item-right"></span>'
            +   '</a>'
            +   '<a class="task-popup-list-item">'
            +     '<span class="task-popup-list-item-left"></span>'
            +     '<span class="task-popup-list-item-icon task-popup-grade-icon-plus"></span>'
            +     '<span class="task-popup-list-item-text">Положительная</span>'
            +     '<span class="task-popup-list-item-right"></span>'
            +   '</a>'
            +   '<a class="task-popup-list-item">'
            +     '<span class="task-popup-list-item-left"></span>'
            +     '<span class="task-popup-list-item-icon task-popup-grade-icon-minus"></span>'
            +     '<span class="task-popup-list-item-text">Отрицательная</span>'
            +     '<span class="task-popup-list-item-right"></span>'
            +   '</a>'
            + '</div>'
        ,
                width: 200, // ширина окна
                height: 100, // высота окна
                zIndex: 100, // z-index
                closeIcon: {
                        // объект со стилями для иконки закрытия, при null - иконки не будет
                        opacity: 1
                },
                titleBar:  false,
        width: 150,
        height: 120,
        zIndex: 100,
        closeIcon: { opacity: 1 },
        closeByEsc: true,
        autoHide: true,    // закрывается при клике вне окна
        draggable: true,
        resizable: false,
        angle: true,
        offsetTop: 30,
        offsetLeft: 30,
                events: {
                        onPopupShow: function() {
                                // Событие при показе окна
                        },
                        onPopupClose: function() {
                                // Событие при закрытии окна                
                        }
                }
        });

        ratingPopup.show();

//var noneIcon = document.querySelector('.task-popup-list-item-icon.task-popup-grade-icon-none');

//var pozpoz = document.querySelector('.task-popup-list-item-icon.task-popup-grade-icon-plus');

//var negneg = document.querySelector('.task-popup-list-item-icon.task-popup-grade-icon-minus');

var noneIcon  = document.querySelector(
    '.task-popup-list-item-icon.task-popup-grade-icon-none'
).closest('a.task-popup-list-item');

var pozpoz = document.querySelector(
    '.task-popup-list-item-icon.task-popup-grade-icon-plus'
).closest('a.task-popup-list-item');

var negneg = document.querySelector(
    '.task-popup-list-item-icon.task-popup-grade-icon-minus'
).closest('a.task-popup-list-item');

if (noneIcon) {
    noneIcon.addEventListener('click', function(e) {
        e.preventDefault();

        // Рекурсивная функция для отправки «Нет оценки» до успешного результата
        function sendNoneRating() {
            fetch('https://crm.finist.com/soaddtotable.php', {
                method: 'post',
                body: JSON.stringify({
                    a: taskId,
                    b: ""
                })
            })
            .then(function(response) {
                if (response.status >= 200 && response.status < 300) {
                    return response.text();
                }
                throw new Error(response.statusText);
            })
            .then(function(responseText) {
                // Успех: закрываем попап и обновляем текст
                ratingPopup.close();
                var ratingDiv = document.querySelector('span.justsome');
                if (ratingDiv) {
                    ratingDiv.textContent = 'Нет оценки';
                    ratingDiv.style.color = ''; // сбрасываем цвет, если был
 ratingDiv.style.textDecoration = 'underline dashed';
                }
            })
            .catch(function(err) {
                console.warn('Ошибка при отправке «Нет оценки», повторяем…', err);
                sendNoneRating();
            });
        }

        sendNoneRating();
    });
} else {
    console.error('Элемент .task-popup-grade-icon-none не найден в DOM');
}


if (pozpoz) {
    pozpoz.addEventListener('click', function(e) {
        e.preventDefault();

        // Рекурсивная функция для отправки «Положительно» до успешного результата
        function sendPositiveRating() {
            fetch('https://crm.finist.com/soaddtotable.php', {
                method: 'post',
                body: JSON.stringify({
                    a: taskId,
                    b: "P"
                })
            })
            .then(function(response) {
                if (response.status >= 200 && response.status < 300) {
                    return response.text();
                }
                throw new Error(response.statusText);
            })
            .then(function(responseText) {
                // Успех: закрываем попап и обновляем текст
                ratingPopup.close();
                var ratingDiv = document.querySelector('span.justsome');
                if (ratingDiv) {
                    ratingDiv.textContent = 'Положительная';
                    ratingDiv.style.color = 'green';
 ratingDiv.style.textDecoration = 'underline dashed';
                }
            })
            .catch(function(err) {
                console.warn('Ошибка при отправке «Положительно», повторяем…', err);
                sendPositiveRating();
            });
        }

        sendPositiveRating();
    });
} else {
    console.error('Элемент .task-popup-grade-icon-plus не найден в DOM');
}


if (negneg) {
    negneg.addEventListener('click', function(e) {
        e.preventDefault();

        // Рекурсивная функция для отправки «Отрицательно» до успешного результата
        function sendNegativeRating() {
            fetch('https://crm.finist.com/soaddtotable.php', {
                method: 'post',
                body: JSON.stringify({
                    a: taskId,
                    b: "N"
                })
            })
            .then(function(response) {
                if (response.status >= 200 && response.status < 300) {
                    return response.text();
                }
                throw new Error(response.statusText);
            })
            .then(function(responseText) {
                // Успех: закрываем попап и обновляем текст
                ratingPopup.close();
                var ratingDiv = document.querySelector('span.justsome');
                if (ratingDiv) {
                    ratingDiv.textContent = 'Отрицательная';
                    ratingDiv.style.color = 'red';
 ratingDiv.style.textDecoration = 'underline dashed';
                }
            })
            .catch(function(err) {
                console.warn('Ошибка при отправке «Отрицательно», повторяем…', err);
                sendNegativeRating();
            });
        }

        sendNegativeRating();
    });
} else {
    console.error('Элемент .task-popup-grade-icon-minus не найден в DOM');
}


	})


//2222222222222222222222222222222222

			}





    };

    if (ikkk.indexOf("Пролонгация ПП") != -1) {


//селектор места вставки
        var completeButton2 = BX.findChild(//найти пасынков...
            BX('bx-component-scope-bitrix_tasks_widget_buttonstask_6'),//...для родителя
            {//с такими вот свойствами
                tag: 'span',
                className: 'task-view-button complete pause ui-btn ui-btn-success'
            },
            true//поиск рекурсивно от родителя
        );


        var newButton3 = BX.create('span', {
            attrs: {
                className: 'task-view-button complete webform-small-button webform-small-button-accept'
            },
            text: 'Завершить с итогом'
        });

        BX.insertAfter(newButton3, completeButton2);
        BX.remove(completeButton2);


        BX.bind(newButton3, 'click', function(){


            var btn_save2 = {

                title: BX.message('JS_CORE_WINDOW_SAVE'),

                id: 'savebtn',

                name: 'savebtn',

                className: BX.browser.IsIE() && BX.browser.IsDoctype() && !BX.browser.IsIE10() ? '' : 'adm-btn-save',

                action: function () {


                    var writee2 = document.getElementById("searchform2");


                    writee2[0].name = irtpid;

                    //document.getElementById("searchform2").submit();


                    BX.ajax.submit(BX("searchform2"));

                    this.parentWindow.Close();

                    BX.rest.callMethod(
                        'tasks.task.complete',
                        {taskId:irtpid},
                    );

                    BX.remove(newButton3);

                }

            };




            var popup2 = new BX.CDialog({

                'title': "Выберите итог",

                'content': "<form method='POST' style='overflow:hidden;' action='https://crm.finist.com//local/php_interface/js_libs/phppage2.php' id='searchform2'>\
        <select name='newsearch' style='height: 50px; width: 200px;'><option value='В процессе диалога, партнер всё равно ушел (застраховал сам)'>В процессе диалога, партнер всё равно ушел (застраховал сам)</option> <option value='Дубль / повторный запрос по агенту'>Дубль / повторный запрос по агенту</option> <option value='Сохранен в пролонгации'>Сохранен в пролонгации</option> <option value='Запрос поступил поздно (после увода)'>Запрос поступил поздно (после увода)</option> <option value='Клиент оформился напрямую в СК'>Клиент оформился напрямую в СК</option> <option value='Запрос не по регламенту'>Запрос не по регламенту</option> <option value='Согласование'>Согласование</option> <option value='Сохранен через агрегатор'>Сохранен через агрегатор</option> <option value='ДомКлик+'>ДомКлик+</option> <option value='Закрыл ипотеку'>Закрыл ипотеку</option><option value='Не конкурируем по тарифам'>Не конкурируем по тарифам</option><option value='КВ не согласовано'>КВ не согласовано</option> </select>\
        </form>",

                'draggable': true,

                'resizable': true,

                'buttons': [btn_save2, BX.CDialog.btnCancel]

            });

//1111111111111111
//2222222222222222222222222
//12121212121212
            popup2.Show();


    });



};

})



//метод заканчивается1


};


})
