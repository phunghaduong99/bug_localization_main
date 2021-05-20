/*******************************************************************************
 * Copyright (c) 2009, 2010 IBM Corporation and others.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * http://www.eclipse.org/legal/epl-v10.html
 *
 * Contributors:
 *     IBM Corporation - initial API and implementation
 ******************************************************************************/

package org.eclipse.e4.ui.tests.reconciler;

import java.util.Collection;
import org.eclipse.e4.ui.model.application.MApplication;
import org.eclipse.e4.ui.model.application.ui.basic.MPart;
import org.eclipse.e4.ui.model.application.ui.basic.MWindow;
import org.eclipse.e4.ui.model.application.ui.basic.impl.BasicFactoryImpl;
import org.eclipse.e4.ui.model.application.ui.menu.MToolBar;
import org.eclipse.e4.ui.model.application.ui.menu.MToolBarElement;
import org.eclipse.e4.ui.model.application.ui.menu.MToolItem;
import org.eclipse.e4.ui.model.application.ui.menu.impl.MenuFactoryImpl;
import org.eclipse.e4.ui.workbench.modeling.ModelDelta;
import org.eclipse.e4.ui.workbench.modeling.ModelReconciler;

public abstract class ModelReconcilerToolBarTest extends ModelReconcilerTest {

	private void testToolBar_Children_Add(MToolBarElement toolItem) {
		MApplication application = createApplication();

		MWindow window = createWindow(application);

		MPart part = BasicFactoryImpl.eINSTANCE.createPart();
		window.getChildren().add(part);

		MToolBar toolBar = MenuFactoryImpl.eINSTANCE.createToolBar();
		part.setToolbar(toolBar);

		saveModel();

		ModelReconciler reconciler = createModelReconciler();
		reconciler.recordChanges(application);

		toolBar.getChildren().add(toolItem);

		Object state = reconciler.serialize();

		application = createApplication();
		window = application.getChildren().get(0);

		part = (MPart) window.getChildren().get(0);
		toolBar = part.getToolbar();

		Collection<ModelDelta> deltas = constructDeltas(application, state);

		assertEquals(0, toolBar.getChildren().size());

		applyAll(deltas);

		assertEquals(1, toolBar.getChildren().size());
		assertEquals(toolItem.getClass(), toolBar.getChildren().get(0)
				.getClass());
	}

	public void testToolBar_Children_Add_ToolBarSeparator() {
		testToolBar_Children_Add(MenuFactoryImpl.eINSTANCE
				.createToolBarSeparator());
	}

	public void testToolBar_Children_Add_DirectToolItem() {
		testToolBar_Children_Add(MenuFactoryImpl.eINSTANCE
				.createDirectToolItem());
	}

	public void testToolBar_Children_Add_HandledToolItem() {
		testToolBar_Children_Add(MenuFactoryImpl.eINSTANCE
				.createHandledToolItem());
	}

	public void testToolBar_Children_Remove() {
		MApplication application = createApplication();

		MWindow window = createWindow(application);

		MPart part = BasicFactoryImpl.eINSTANCE.createPart();
		window.getChildren().add(part);

		MToolBar toolBar = MenuFactoryImpl.eINSTANCE.createToolBar();
		part.setToolbar(toolBar);

		MToolItem toolItem = MenuFactoryImpl.eINSTANCE.createDirectToolItem();
		toolBar.getChildren().add(toolItem);

		saveModel();

		ModelReconciler reconciler = createModelReconciler();
		reconciler.recordChanges(application);

		toolBar.getChildren().remove(0);

		Object state = reconciler.serialize();

		application = createApplication();
		window = application.getChildren().get(0);

		part = (MPart) window.getChildren().get(0);
		toolBar = part.getToolbar();
		toolItem = (MToolItem) toolBar.getChildren().get(0);

		Collection<ModelDelta> deltas = constructDeltas(application, state);

		assertEquals(1, toolBar.getChildren().size());
		assertEquals(toolItem, toolBar.getChildren().get(0));

		applyAll(deltas);

		assertEquals(0, toolBar.getChildren().size());
	}
}
